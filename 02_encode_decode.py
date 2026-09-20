from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

text = "Data Engineering is interesting."

# Convert text into token IDs
encoded = tokenizer.encode(text)

print("Original text:", text)
print("Token IDs:", encoded)

# Convert token IDs back into text
decoded = tokenizer.decode(encoded)

print("Decoded text:", decoded)