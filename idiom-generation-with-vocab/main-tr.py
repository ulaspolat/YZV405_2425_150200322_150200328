import os
import time
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ChatGPT API Configuration
API_KEY = os.getenv("OPENAI_API_KEY")  # OpenAI API key
MODEL_ID = "gpt-4.1-mini"  # Updated to GPT-4.1-mini

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Define rate limits - GPT-4.1-mini için değerler
RPM = 500  # Requests per minute (GPT-4.1-mini için 500 RPM)
RPD = 10000  # Requests per day (Approximate value based on RPM)
TPM = 200000  # Tokens per minute (GPT-4.1-mini için 200.000 TPM)
TPD = 2000000  # Tokens per day (GPT-4.1-mini için 2.000.000 TPD)
BATCH_SIZE = 10  # Daha yüksek bir batch boyutu kullanabiliriz - rate limiti yüksek olduğu için

# Rate limiti takip etmek için değişkenler
last_request_time = 0
requests_in_current_minute = 0
tokens_in_current_minute = 0
requests_today = 0 
tokens_today = 0
current_minute_start = 0
current_day_start = 0

# Her deyim için kaç örnek üretileceğini belirle
IDIOMATIC_EXAMPLES_PER_IDIOM = 120 #120 yapıyordu
LITERAL_EXAMPLES_PER_IDIOM = 120 #120 yapıyordu

# Bir seferde kaç örnek isteneceğini belirle (API yanıt limitleri nedeniyle)
EXAMPLES_PER_REQUEST_IDIOMATIC = 30  # Bir API çağrısında istenen idiomatic örnek sayısı 30 yapıyordu
EXAMPLES_PER_REQUEST_LITERAL = 30  # Bir API çağrısında istenen literal örnek sayısı 30 yapıyordu

# Path setup
DATASET_DIR = Path("dataset/tr")
DATASET_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = DATASET_DIR / "idioms_data.json"
IDIOMS_FILE = Path("idioms/tr-idioms.json")

# Load Turkish idioms from JSON file
def load_idioms(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        idioms_data = json.load(f)
    
    # Transform data to the format needed for prompt creation
    idioms_list = []
    for idiom, meanings in idioms_data.items():
        idioms_list.append({
            "deyim": idiom,
            "mecaz anlamı": meanings.get("mecaz_anlam", ""),
            "gerçek anlamı": meanings.get("gerçek_anlam", "")
        })
    
    return idioms_list

def create_prompt_idiomatic(idiom_data, num_examples=EXAMPLES_PER_REQUEST_IDIOMATIC):
    return f"""Sen bir Türkçe dil uzmanısın ve dilimizi zengin, yaratıcı biçimlerde kullanmalısın. 
    Lütfen cümlelerinde benzer yapıların tekrarı ve insan isimlerinin kullanımı gibi kalıplardan kaçın. 

Görevin, aşağıdaki deyimin mecaz (idiomatic) anlamında kullanıldığı {num_examples} adet örnek cümle oluşturmak:

Deyim: {idiom_data['deyim']}  
Mecaz anlamı: {idiom_data['mecaz anlamı']}  

Her cümle için şu bilgileri JSON formatında döndür:
1. "sentence": Oluşturduğun cümle
2. "tokenized_sentence": Cümlenin kelimelere ve noktalama işaretlerine ayrılmış hali (dizi formatında)
3. "expression": Kullanılan deyim (örn: "{idiom_data['deyim']}")
4. "category": "idiomatic"
5. "indices": Deyimin cümlede geçtiği haliyle (çekimli ya da ekli olabilir) tokenized_sentence içindeki tüm kelimelerinin sıfırdan başlayan indekslerini bir dizi olarak belirt (örneğin: [3, 4, 5]). Deyimi oluşturan tüm kelimelerin indekslerini tek tek içermelidir.


Kurallar:
- Deyimin tüm kelimelerini cümlede kullan
- Farklı cümle yapıları tercih et (olumlu, olumsuz, soru, emir, vs.)
- Farklı bağlamlarda kullan (resmi, gayriresmi, mizahi, akademik vs.)
- Deyimleri doğal çekimli halleriyle kullan
- En az {num_examples} farklı cümle üret
- Cümlenin anlamlı ve düzgün olduğundan emin ol

SADECE aşağıdaki JSON formatında yanıt ver. Ekstra açıklama veya metin ekleme:

```json
[
  {{
    "sentence": "Buraya deyim içeren, cümle yazılacak.",
    "tokenized_sentence": ["Buraya", "deyim", "içeren", ",", "cümle", "yazılacak", "."],
    "expression": "{idiom_data['deyim']}",
    "category": "idiomatic",
    "indices": [1, 2]
  }},
  ...
]
```

ÖNEMLI: Yanıtın JSON olarak ayrıştırılabilir olduğundan emin ol. Sadece yukarıdaki formatta JSON yap. Ekstra açıklama veya ek bilgi ekleme."""

def create_prompt_literal(idiom_data, num_examples=EXAMPLES_PER_REQUEST_LITERAL):
    return f"""Sen bir Türkçe dil uzmanısın ve dilimizi zengin, yaratıcı biçimlerde kullanmalısın. 
    Lütfen cümlelerinde benzer yapıların tekrarı ve insan isimlerinin kullanımı gibi kalıplardan kaçın. 

Görevin, aşağıdaki deyimin gerçek (literal) anlamında kullanıldığı {num_examples} adet örnek cümle oluşturmak:

Deyim: {idiom_data['deyim']}  
Gerçek anlamı: {idiom_data['gerçek anlamı']}  

Her cümle için şu bilgileri JSON formatında döndür:
1. "sentence": Oluşturduğun cümle
2. "tokenized_sentence": Cümlenin kelimelere ve noktalama işaretlerine ayrılmış hali (dizi formatında)
3. "expression": Kullanılan deyim (örn: "{idiom_data['deyim']}")
4. "category": "literal"
5. "indices": [-1]

Kurallar:
- Deyimin tüm kelimelerini cümlede kullan
- Farklı cümle yapıları tercih et (olumlu, olumsuz, soru, emir, vs.)
- Farklı bağlamlarda kullan (resmi, gayriresmi, mizahi, akademik vs.)
- Deyimleri doğal çekimli halleriyle kullan
- En az {num_examples} farklı cümle üret
- Cümlenin anlamlı ve düzgün olduğundan emin ol

SADECE aşağıdaki JSON formatında yanıt ver. Ekstra açıklama veya metin ekleme:

```json
[
  {{
    "sentence": "Buraya deyim içeren, cümle yazılacak.",
    "tokenized_sentence": ["Buraya", "deyim", "içeren", ",", "cümle", "yazılacak", "."],
    "expression": "{idiom_data['deyim']}",
    "category": "literal",
    "indices": [-1] 
  }},
  ...
]
```

ÖNEMLI: Yanıtın JSON olarak ayrıştırılabilir olduğundan emin ol. Sadece yukarıdaki formatta JSON yap. Ekstra açıklama veya ek bilgi ekleme."""

def call_api(prompt):
    """Call the OpenAI API with rate limiting"""
    global last_request_time, requests_in_current_minute, tokens_in_current_minute
    global requests_today, tokens_today, current_minute_start, current_day_start
    
    url = "https://api.openai.com/v1/chat/completions"
    
    data = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 4000
    }
    
    # Tahmini token sayısı (basit hesaplama: karakter sayısı / 4)
    estimated_tokens = len(prompt) // 4 + 4000  # prompt + max yanıt tokenleri
    
    current_time = time.time()
    
    # Gün ve dakika kontrolü
    if current_day_start == 0:
        current_day_start = current_time
    elif current_time - current_day_start > 86400:  # 24 saat geçtiyse
        current_day_start = current_time
        requests_today = 0
        tokens_today = 0
        print("Daily rate limit counters reset.")
    
    if current_minute_start == 0:
        current_minute_start = current_time
    elif current_time - current_minute_start > 60:  # 1 dakika geçtiyse
        current_minute_start = current_time
        requests_in_current_minute = 0
        tokens_in_current_minute = 0
        print("Minute rate limit counters reset.")
    
    # Dakika rate limit kontrolü
    if requests_in_current_minute >= RPM:
        wait_time = 60 - (current_time - current_minute_start)
        if wait_time > 0:
            print(f"Minute request rate limit ({RPM} RPM) reached. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
            # Sayaçları sıfırla
            current_minute_start = time.time()
            requests_in_current_minute = 0
            tokens_in_current_minute = 0
    
    if tokens_in_current_minute + estimated_tokens >= TPM:
        wait_time = 60 - (current_time - current_minute_start)
        if wait_time > 0:
            print(f"Minute token rate limit ({TPM} TPM) reached. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
            # Sayaçları sıfırla
            current_minute_start = time.time()
            requests_in_current_minute = 0
            tokens_in_current_minute = 0
    
    # Gün rate limit kontrolü
    if requests_today >= RPD:
        wait_time = 86400 - (current_time - current_day_start)
        if wait_time > 0:
            print(f"Daily request rate limit ({RPD} RPD) reached. Waiting {wait_time/3600:.1f} hours...")
            time.sleep(min(wait_time, 3600))  # En fazla 1 saat bekle, sonra tekrar kontrol et
            return call_api(prompt)  # Recursive call to retry
    
    if tokens_today + estimated_tokens >= TPD:
        wait_time = 86400 - (current_time - current_day_start)
        if wait_time > 0:
            print(f"Daily token rate limit ({TPD} TPD) reached. Waiting {wait_time/3600:.1f} hours...")
            time.sleep(min(wait_time, 3600))  # En fazla 1 saat bekle, sonra tekrar kontrol et
            return call_api(prompt)  # Recursive call to retry
    
    # İstekler arası minimum bekleme süresi (rate limit aşımını önlemek için)
    time_since_last_request = current_time - last_request_time
    if time_since_last_request < 0.12:  # 500 RPM için minimum 0.12 saniye bekleme
        time.sleep(0.12 - time_since_last_request)
    
    # Request yapma
    try:
        last_request_time = time.time()
        requests_in_current_minute += 1
        requests_today += 1
        tokens_in_current_minute += estimated_tokens
        tokens_today += estimated_tokens
        
        response = requests.post(url, headers=HEADERS, json=data)
        
        if response.status_code == 200:
            # Kullanılan gerçek token sayısını güncelle
            if 'usage' in response.json():
                usage = response.json()['usage']
                real_tokens = usage.get('total_tokens', estimated_tokens)
                # Tahmini değerleri düzelt
                tokens_in_current_minute = tokens_in_current_minute - estimated_tokens + real_tokens
                tokens_today = tokens_today - estimated_tokens + real_tokens
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            if response.status_code == 429:  # Rate limit exceeded
                retry_after = int(response.headers.get('Retry-After', 60))
                print(f"Rate limit exceeded. Waiting {retry_after} seconds.")
                time.sleep(retry_after)
                # Sayaçları sıfırla
                current_minute_start = time.time()
                requests_in_current_minute = 0
                tokens_in_current_minute = 0
                return call_api(prompt)  # Retry after waiting
            elif response.status_code == 400:
                print("Bad request. Check the API parameters.")
            elif response.status_code == 401:
                print("Unauthorized. Check your API key.")
            elif response.status_code == 404:
                print("Model not found. Check the model ID.")
            return None
    except Exception as e:
        print(f"Exception occurred: {e}")
        return None

def parse_response(response):
    """Parse the LLM response to extract examples from JSON format"""
    if not response:
        print("Empty response received")
        return []
    
    # API yanıtını kontrol et
    if 'choices' not in response:
        print(f"Invalid response format: 'choices' field missing. Response structure: {str(response)[:200]}...")
        return []
    
    try:
        # Yanıt içeriğini al
        if len(response['choices']) == 0:
            print("Empty 'choices' array in response")
            return []
        
        # message içeriğini almaya çalış
        if 'message' in response['choices'][0]:
            content = response['choices'][0]['message'].get('content', '')
        else:
            content = response['choices'][0].get('text', '')
            if not content:
                print(f"Could not extract content from response: {str(response['choices'][0])[:200]}...")
                return []
    except (KeyError, IndexError) as e:
        print(f"Error accessing response content: {e}")
        print(f"Response structure: {str(response)[:200]}...")
        return []
    
    # JSON bloğunu çıkarmak için
    json_content = content.strip()
    
    # Markdown kod bloğu formatındaysa çıkar
    if "```json" in json_content and "```" in json_content:
        # Başlangıç ve bitiş işaretlerini bul
        start_marker = "```json"
        end_marker = "```"
        start_idx = json_content.find(start_marker) + len(start_marker)
        end_idx = json_content.rfind(end_marker)
        if start_idx != -1 and end_idx != -1:
            json_content = json_content[start_idx:end_idx].strip()
    # Eğer sadece ``` ile başlıyorsa (dil belirtilmemiş)
    elif json_content.strip().startswith("```") and "```" in json_content[3:]:
        start_idx = json_content.find("```") + 3
        end_idx = json_content.rfind("```")
        if start_idx != -1 and end_idx != -1:
            json_content = json_content[start_idx:end_idx].strip()
    
    # JSON formatında değilse ancak [ ile başlayıp ] ile bitiyorsa
    if not (json_content.strip().startswith("[") and json_content.strip().endswith("]")):
        # İçerikte [ ] arasındaki kısmı bulmaya çalış
        start_idx = json_content.find("[")
        end_idx = json_content.rfind("]") + 1  # +1 to include ]
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            json_content = json_content[start_idx:end_idx].strip()
    
    try:
        # JSON'ı ayrıştır
        examples_data = json.loads(json_content)
        
        # Sonuçları oluştur
        examples = []
        for item in examples_data:
            if isinstance(item, dict):  # item bir dictionary mi kontrol et
                examples.append({
                    "sentence": item.get("sentence", ""),
                    "tokenized_sentence": item.get("tokenized_sentence", []),
                    "expression": item.get("expression", ""),
                    "category": item.get("category", ""),
                    "indices": item.get("indices", [])
                })
        
        # Boş alanları filtrele
        examples = [ex for ex in examples if ex["sentence"] and ex["tokenized_sentence"] and ex["expression"]]
        
        if not examples:
            print("JSON parsed successfully but no valid examples found in the data")
            print(f"Parsed JSON content: {json_content[:300]}...")
        
        return examples
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Content was: {json_content[:300]}...")
        return []
    except Exception as e:
        print(f"Error processing JSON response: {e}")
        print(f"Content was: {content[:300]}...")
        return []

def write_to_json(examples, output_file):
    """Write examples to JSON file"""
    # Load existing data if file exists
    data = []
    try:
        if Path(output_file).exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # If file doesn't exist or is invalid JSON, start with empty list
        data = []
    
    # Get the current max ID
    current_id = 0
    if data:
        current_id = max(item.get('id', 0) for item in data)
    
    # Add new examples
    for example in examples:
        current_id += 1
        data.append({
            'id': current_id,
            'language': 'tr',
            'sentence': example['sentence'],
            'tokenized_sentence': example['tokenized_sentence'],
            'expression': example['expression'],
            'category': example['category'],
            'indices': example['indices']
        })
    
    # Write all data back to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def generate_examples(idiom_data, category, total_examples):
    """Generate examples for an idiom with the given category in smaller batches"""
    all_examples = []
    examples_per_request = EXAMPLES_PER_REQUEST_IDIOMATIC if category == "idiomatic" else EXAMPLES_PER_REQUEST_LITERAL
    
    # Kaç istek yapılacağını hesapla
    num_requests = (total_examples + examples_per_request - 1) // examples_per_request
    
    for i in range(num_requests):
        # Son istek için kalan örnekleri hesapla
        if i == num_requests - 1 and total_examples % examples_per_request != 0:
            remaining = total_examples % examples_per_request
            current_request_size = remaining
        else:
            current_request_size = examples_per_request
        
        print(f"  Generating {category} examples batch {i+1}/{num_requests} ({current_request_size} examples) for '{idiom_data['deyim']}'")
        
        # Prompt oluştur
        if category == "idiomatic":
            prompt = create_prompt_idiomatic(idiom_data, current_request_size)
        else:
            prompt = create_prompt_literal(idiom_data, current_request_size)
        
        # Retry mechanism for API calls
        max_retries = 3
        retry_count = 0
        examples = []
        
        while retry_count < max_retries and len(examples) < current_request_size:
            # Call API
            response = call_api(prompt)
            
            if response:
                # Parse examples
                batch_examples = parse_response(response)
                
                if batch_examples:
                    examples.extend(batch_examples)
                    print(f"    Generated {len(batch_examples)} {category} examples")
                
                # Yeterli örnek yoksa tekrar deneyelim
                if len(examples) < current_request_size and retry_count < max_retries - 1:
                    retry_count += 1
                    remaining = current_request_size - len(examples)
                    print(f"    Only got {len(examples)}/{current_request_size} examples. Generating {remaining} more...")
                    
                    # Kalan örnekler için yeni prompt oluştur
                    if category == "idiomatic":
                        prompt = create_prompt_idiomatic(idiom_data, remaining)
                    else:
                        prompt = create_prompt_literal(idiom_data, remaining)
                elif len(examples) < current_request_size:
                    print(f"    Could not generate all {current_request_size} {category} examples after {retry_count+1} attempts. Got {len(examples)}.")
            else:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"    API call failed. Retrying ({retry_count}/{max_retries}) after 5 seconds...")
                    time.sleep(5)  # API hata verdiğinde daha kısa süre bekle (rate limit değilse)
                else:
                    print(f"    API call failed after {max_retries} attempts for '{idiom_data['deyim']}'.")
        
        # Add examples to all_examples
        all_examples.extend(examples)
        print(f"    Total {category} examples so far: {len(all_examples)}/{total_examples}")
        
        # İstekler arası rate limit için artık burada bekleme yapmıyoruz
        # Rate limit kontrolü call_api fonksiyonu içine taşındı
    
    # Örnekleri geri döndür
    return all_examples[:total_examples]  # Eğer fazla örnek üretildiyse, sadece istenilen sayıda örneği döndür

def main():
    # Check if API key exists
    if not API_KEY:
        print("Error: OPENAI_API_KEY not found. Please set it in your .env file.")
        return
    
    # Check if idioms file exists
    if not IDIOMS_FILE.exists():
        print(f"Error: Idioms file not found at {IDIOMS_FILE}")
        return
        
    # Load idioms from JSON file
    tr_idioms = load_idioms(IDIOMS_FILE)
    print(f"Loaded {len(tr_idioms)} Turkish idioms from {IDIOMS_FILE}")
        
    print(f"Starting generation for {len(tr_idioms)} Turkish idioms using GPT-4.1-mini...")
    print(f"Each idiom will have {IDIOMATIC_EXAMPLES_PER_IDIOM} idiomatic and {LITERAL_EXAMPLES_PER_IDIOM} literal examples")
    print(f"Processing in batches of {BATCH_SIZE} idioms...")
    
    # Process idioms in batches
    for i in range(0, len(tr_idioms), BATCH_SIZE):
        batch = tr_idioms[i:i+BATCH_SIZE]
        print(f"\nStarting batch {i//BATCH_SIZE + 1}/{(len(tr_idioms) + BATCH_SIZE - 1) // BATCH_SIZE}")
        
        for idx, idiom_data in enumerate(batch):
            print(f"\nProcessing idiom {i+idx+1}/{len(tr_idioms)}: {idiom_data['deyim']}")
            
            all_examples = []
            
            # Generate idiomatic examples in batches
            print(f"Generating {IDIOMATIC_EXAMPLES_PER_IDIOM} idiomatic examples for '{idiom_data['deyim']}'")
            idiomatic_examples = generate_examples(idiom_data, "idiomatic", IDIOMATIC_EXAMPLES_PER_IDIOM)
            
            # Add idiomatic examples to all_examples
            if idiomatic_examples:
                all_examples.extend(idiomatic_examples)
                print(f"Successfully generated {len(idiomatic_examples)} idiomatic examples")
            
            # Generate literal examples in batches
            print(f"Generating {LITERAL_EXAMPLES_PER_IDIOM} literal examples for '{idiom_data['deyim']}'")
            literal_examples = generate_examples(idiom_data, "literal", LITERAL_EXAMPLES_PER_IDIOM)
            
            # Add literal examples to all_examples
            if literal_examples:
                all_examples.extend(literal_examples)
                print(f"Successfully generated {len(literal_examples)} literal examples")
            
            # Write all examples to JSON
            if all_examples:
                write_to_json(all_examples, OUTPUT_FILE)
                print(f"Added total {len(all_examples)} examples for '{idiom_data['deyim']}' ({sum(1 for e in all_examples if e['category'].lower() == 'idiomatic')} idiomatic, {sum(1 for e in all_examples if e['category'].lower() == 'literal')} literal)")
            else:
                print(f"No valid examples generated for '{idiom_data['deyim']}'.")

    print(f"\nData generation complete. Results saved to {OUTPUT_FILE} in JSON format")

if __name__ == "__main__":
    main()
