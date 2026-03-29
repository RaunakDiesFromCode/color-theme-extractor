from theme_extractor import ThemeExtractor

extractor = ThemeExtractor()

result = extractor.extract("data/test.png")
theme = result["theme"]

print("\nGenerated Theme:\n")
for k, v in theme.items():
    print(f"{k}: {v}")

print("\nAccent Color:", result["accent_rgb"])
print("Palette:", result["palette_rgb"])
