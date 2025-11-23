"""Check actual vocab sizes of different Speech2Text tokenizers"""
from transformers import Speech2TextTokenizer

# Check original English model tokenizer
print("Checking tokenizers vocab sizes...")
print("="*60)

english_tokenizer = Speech2TextTokenizer.from_pretrained("facebook/s2t-small-librispeech-asr")
print(f"English (s2t-small-librispeech-asr): {len(english_tokenizer)}")
print(f"  - Has lang_code_to_id: {hasattr(english_tokenizer, 'lang_code_to_id')}")
if hasattr(english_tokenizer, 'lang_code_to_id'):
    print(f"  - Languages: {list(english_tokenizer.lang_code_to_id.keys())}")

multilingual_tokenizer = Speech2TextTokenizer.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")
print(f"\nMultilingual (s2t-medium-mustc-multilingual-st): {len(multilingual_tokenizer)}")
print(f"  - Has lang_code_to_id: {hasattr(multilingual_tokenizer, 'lang_code_to_id')}")
if hasattr(multilingual_tokenizer, 'lang_code_to_id'):
    langs = list(multilingual_tokenizer.lang_code_to_id.keys())
    print(f"  - Languages ({len(langs)}): {langs}")
    print(f"  - Russian supported: {'ru' in langs}")

print("="*60)
print(f"Vocab sizes are {'SAME' if len(english_tokenizer) == len(multilingual_tokenizer) else 'DIFFERENT'}")
