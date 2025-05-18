# Idiom Generation Dataset

This repository contains scripts to generate synthetic datasets for idiom identification in Turkish and Italian languages.

## Setup

1. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the root directory with your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. Ensure the `/dataset` directory exists with subdirectories for each language:
   ```
   /dataset
     /tr
     /it
   ```

4. Make sure the idioms JSON files are available:
   ```
   /idioms
     tr-idioms.json
     it-idioms.json
   ```

## JSON Format

The idioms are stored in JSON files with the following format:

**Turkish (tr-idioms.json):**
```json
{
  "idiom expression": {
    "mecaz_anlam": "Idiomatic meaning",
    "gerçek_anlam": "Literal meaning"
  }
}
```

**Italian (it-idioms.json):**
```json
{
  "idiom expression": {
    "significato_idiomatico": "Idiomatic meaning",
    "significato_letterale": "Literal meaning"
  }
}
```

## Usage

### Generate Turkish Idiom Examples
```
python main-tr.py
```

### Generate Italian Idiom Examples
```
python main-it.py
```

## Dataset Format

The generated CSV files (`dataset/tr/idioms_data.csv` and `dataset/it/idioms_data.csv`) will have the following format:

```
id,language,sentence,tokenized_sentence,expression,category,indices
1,tr,"Example sentence",["tokenized","sentence"],idiom expression,idiomatic/literal,[indices]
```

- `id`: Unique identifier for each example
- `language`: Language code ("tr" for Turkish, "it" for Italian)
- `sentence`: Full sentence containing an idiom
- `tokenized_sentence`: The sentence split into tokens
- `expression`: The idiom expression found in the sentence
- `category`: Whether the idiom is used in its "idiomatic" or "literal" sense
- `indices`: Token indices for the idiom in the sentence ([-1] for literal uses)

## Rate Limits

The scripts respect Groq's API rate limits (30 requests per minute, 1000 requests per day) by:
- Processing idioms in batches
- Adding delays between API calls
- Handling rate limit errors gracefully

## Customization

You can customize the idioms by modifying the JSON files in the `idioms` directory. 