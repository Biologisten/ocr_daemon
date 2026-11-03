#include "ocr_core.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {
void send_response(int client, int status, const std::string &body) {
  std::string reason = status == 200 ? "OK" : status == 400 ? "Bad Request" : "Internal Server Error";
  std::string head = "HTTP/1.1 " + std::to_string(status) + " " + reason + "\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: " + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n";
  send(client, head.data(), head.size(), MSG_NOSIGNAL);
  send(client, body.data(), body.size(), MSG_NOSIGNAL);
}

bool read_request(int client, std::vector<unsigned char> &body) {
  std::string request;
  char buffer[4096];
  while (request.find("\r\n\r\n") == std::string::npos) {
    auto got = recv(client, buffer, sizeof(buffer), 0);
    if (got <= 0 || request.size() + got > 1024 * 1024) return false;
    request.append(buffer, got);
  }
  auto end = request.find("\r\n\r\n"), length_at = request.find("Content-Length:");
  if (request.rfind("POST ", 0) != 0 || length_at == std::string::npos) return false;
  std::size_t length = std::strtoull(request.c_str() + length_at + 15, nullptr, 10);
  if (length > 32 * 1024 * 1024) return false;
  body.assign(request.begin() + end + 4, request.end());
  while (body.size() < length) {
    auto got = recv(client, buffer, std::min(sizeof(buffer), length - body.size()), 0);
    if (got <= 0) return false;
    body.insert(body.end(), buffer, buffer + got);
  }
  body.resize(length);
  return true;
}
}  // namespace

int main() {
  const char *model_dir = std::getenv("OCR_MODEL_DIR");
  if (!model_dir) { std::cerr << "OCR_MODEL_DIR is required\n"; return 2; }
  char *error = nullptr;
  void *engine = ocr_create(model_dir, &error);
  if (!engine) { std::cerr << error << '\n'; ocr_free(error); return 2; }
  int server = socket(AF_INET, SOCK_STREAM, 0), reuse = 1;
  setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  int port = std::getenv("OCR_PORT") ? std::atoi(std::getenv("OCR_PORT")) : 9722;
  sockaddr_in address{}; address.sin_family = AF_INET; address.sin_port = htons(port); address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (bind(server, reinterpret_cast<sockaddr *>(&address), sizeof(address)) || listen(server, 16)) { perror("listen"); return 2; }
  for (;;) {
    int client = accept(server, nullptr, nullptr); if (client < 0) continue;
    std::vector<unsigned char> body;
    if (!read_request(client, body)) send_response(client, 400, "invalid request\n");
    else {
      error = nullptr; char *text = ocr_run(engine, body.data(), body.size(), &error);
      if (text) { send_response(client, 200, text); ocr_free(text); }
      else { send_response(client, 400, error ? error : "ocr failed"); ocr_free(error); }
    }
    close(client);
  }
}
