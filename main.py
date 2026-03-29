from theme_extractor import ThemeExtractor

extractor = ThemeExtractor()

theme = extractor.extract("data/test.png")

print("\nGenerated Theme:\n")
for k, v in theme.items():
    print(f"{k}: {v}")
