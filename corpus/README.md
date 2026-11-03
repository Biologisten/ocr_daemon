# Benchmark corpus

Add exactly 20 sanitized PNG or JPEG screenshots beside `manifest.json`, then
replace `images` with entries of this form:

```json
{
  "file": "01.png",
  "text": "Expected OCR output\nwith one line per detected text region"
}
```

The text is manually transcribed ground truth. Do not include private account
names, messages, tokens, or other sensitive material. The benchmark refuses to
run until there are exactly 20 entries and every named file exists.
