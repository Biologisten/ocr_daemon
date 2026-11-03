#define _POSIX_C_SOURCE 200809L

#include "ocr_core.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#define MAX_HEADER (1024 * 1024)
#define MAX_BODY (32 * 1024 * 1024)

static int send_all(int client, const char *data, size_t size) {
  while (size > 0) {
    ssize_t sent = send(client, data, size, MSG_NOSIGNAL);
    if (sent <= 0) return -1;
    data += sent;
    size -= (size_t)sent;
  }
  return 0;
}

static void send_response(int client, int status, const char *body) {
  const char *reason = status == 200 ? "OK" : status == 400 ? "Bad Request" : "Internal Server Error";
  char head[256];
  int length = snprintf(head, sizeof head,
      "HTTP/1.1 %d %s\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
      status, reason, strlen(body));
  if (length <= 0 || (size_t)length >= sizeof head) return;
  if (send_all(client, head, (size_t)length) == 0) send_all(client, body, strlen(body));
}

struct buffer {
  char *data;
  size_t size;
  size_t capacity;
};

static void buffer_free(struct buffer *buffer) {
  free(buffer->data);
  buffer->data = NULL;
  buffer->size = buffer->capacity = 0;
}

static ssize_t find_bytes(const char *haystack, size_t size, const char *needle, size_t needle_size) {
  if (needle_size == 0 || size < needle_size) return -1;
  for (size_t index = 0; index + needle_size <= size; ++index) {
    if (memcmp(haystack + index, needle, needle_size) == 0) return (ssize_t)index;
  }
  return -1;
}

// Appends one recv() worth of bytes, growing the buffer as needed.
static int receive_more(int client, struct buffer *buffer) {
  if (buffer->capacity - buffer->size < 4096) {
    size_t capacity = buffer->capacity ? buffer->capacity * 2 : 16384;
    char *grown = realloc(buffer->data, capacity);
    if (!grown) return -1;
    buffer->data = grown;
    buffer->capacity = capacity;
  }
  ssize_t got = recv(client, buffer->data + buffer->size, buffer->capacity - buffer->size, 0);
  if (got <= 0) return -1;
  buffer->size += (size_t)got;
  return 0;
}

// Reads one request into *body (malloc'd, caller frees). Returns 0 on success.
static int read_request(int client, struct buffer *buffer, unsigned char **body, size_t *body_size) {
  static const char separator[] = "\r\n\r\n";
  ssize_t end = -1;
  while (end < 0) {
    if (buffer->size > MAX_HEADER) return -1;
    if (receive_more(client, buffer) != 0) return -1;
    end = find_bytes(buffer->data, buffer->size, separator, sizeof separator - 1);
    if (buffer->size > MAX_HEADER) return -1;
  }
  size_t header_end = (size_t)end + sizeof separator - 1;
  if (buffer->size < sizeof "POST " - 1 || memcmp(buffer->data, "POST ", sizeof "POST " - 1) != 0) return -1;
  static const char length_header[] = "Content-Length:";
  ssize_t length_at = find_bytes(buffer->data, (size_t)end, length_header, sizeof length_header - 1);
  if (length_at < 0) return -1;
  char *length_text = buffer->data + length_at + sizeof length_header - 1;
  char *length_end = NULL;
  unsigned long long length = strtoull(length_text, &length_end, 10);
  if (length_end == length_text || length > MAX_BODY) return -1;

  unsigned char *payload = malloc(length ? (size_t)length : 1);
  if (!payload) return -1;
  size_t present = buffer->size - header_end;
  if (present > length) present = (size_t)length;
  memcpy(payload, buffer->data + header_end, present);
  while (present < length) {
    ssize_t got = recv(client, payload + present, (size_t)length - present, 0);
    if (got <= 0) {
      free(payload);
      return -1;
    }
    present += (size_t)got;
  }
  *body = payload;
  *body_size = (size_t)length;
  return 0;
}

int main(void) {
  const char *model_dir = getenv("OCR_MODEL_DIR");
  if (!model_dir) {
    fprintf(stderr, "OCR_MODEL_DIR is required\n");
    return 2;
  }
  char *error = NULL;
  void *engine = ocr_create(model_dir, &error);
  if (!engine) {
    fprintf(stderr, "%s\n", error ? error : "ocr_create failed");
    ocr_free(error);
    return 2;
  }

  int server = socket(AF_INET, SOCK_STREAM, 0);
  if (server < 0) {
    perror("socket");
    return 2;
  }
  int reuse = 1;
  setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof reuse);
  const char *port_text = getenv("OCR_PORT");
  struct sockaddr_in address;
  memset(&address, 0, sizeof address);
  address.sin_family = AF_INET;
  address.sin_port = htons((uint16_t)(port_text ? atoi(port_text) : 9722));
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (bind(server, (struct sockaddr *)&address, sizeof address) != 0 || listen(server, 16) != 0) {
    perror("listen");
    return 2;
  }

  for (;;) {
    int client = accept(server, NULL, NULL);
    if (client < 0) continue;
    struct buffer buffer = {NULL, 0, 0};
    unsigned char *body = NULL;
    size_t body_size = 0;
    if (read_request(client, &buffer, &body, &body_size) != 0) {
      send_response(client, 400, "invalid request\n");
    } else {
      error = NULL;
      char *text = ocr_run(engine, body, body_size, &error);
      if (text) {
        send_response(client, 200, text);
        ocr_free(text);
      } else {
        send_response(client, 400, error ? error : "ocr failed");
        ocr_free(error);
      }
    }
    free(body);
    buffer_free(&buffer);
    close(client);
  }
}
