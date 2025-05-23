# Idiom Generation Dataset

This repository contains scripts to generate synthetic datasets for idiom identification in Turkish and Italian languages. The generated datasets are used for training and evaluating idiom detection models.

## Overview

The project creates a dataset of sentences containing idioms used in either their idiomatic (figurative) or literal sense. Each dataset includes:
- Full sentences containing idioms
- Tokenized versions of those sentences
- Indices of idiom tokens in the sentence
- Classification of whether the idiom is used in its idiomatic or literal sense

## Setup

1. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the root directory with your API key:
   ```
   OPENAI_API_KEY=your_api_key_here
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

## Idiom JSON Format

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
    "significato_fig": "Idiomatic meaning",
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

### Convert JSON to CSV Format
```
python convert_json_to_csv.py
```

## Dataset Generation Process

The data generation process:

1. Loads idioms from JSON files
2. For each idiom, generates examples where the idiom is used in:
   - Its idiomatic/figurative sense (with idiom token indices marked)
   - Its literal sense (with indices set to [-1])
3. Uses GPT-4.1-mini to generate natural sentences for both categories
4. Each generation includes proper tokenization and index tracking
5. Outputs data in JSON format, which can be converted to CSV

## Dataset Format

The generated CSV files (`dataset/tr/idioms_data.csv` and `dataset/it/idioms_data.csv`) have the following format:

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
- `indices`: Token indices for the idiom in the sentence
  - For idiomatic use: Actual indices of idiom tokens [3, 4, 5]
  - For literal use: [-1]

## Rate Limiting

The scripts implement sophisticated rate limiting to respect API constraints:
- Processes idioms in batches
- Tracks requests per minute (RPM) and tokens per minute (TPM)
- Handles daily rate limits
- Implements retry mechanisms for failed API calls
- Dynamically adjusts wait times between API calls

## Customization

You can customize the dataset generation by:
- Modifying the idioms in the JSON files
- Adjusting the number of examples per idiom in the main scripts:
  ```python
  IDIOMATIC_EXAMPLES_PER_IDIOM = 120
  LITERAL_EXAMPLES_PER_IDIOM = 120
  ```
- Changing the batch size and API parameters in the main scripts

