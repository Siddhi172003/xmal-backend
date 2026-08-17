import requests

MYMEMORY_URL = "https://api.mymemory.translated.net/get"


def split_text(text, max_chars=450):
    """
    Split text into chunks while keeping words intact.
    450 is used instead of 500 to stay safely below the limit.
    """

    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:

        if len(current_chunk) + len(word) + 1 <= max_chars:
            if current_chunk:
                current_chunk += " "
            current_chunk += word

        else:
            if current_chunk:
                chunks.append(current_chunk)

            current_chunk = word

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def translate_text(text, target_language, source_language="en"):

    if not text:
        return text

    if target_language == "en":
        return text

    try:

        # Short text: use a single request
        if len(text) <= 450:

            response = requests.get(
                MYMEMORY_URL,
                params={
                    "q": text,
                    "langpair": f"{source_language}|{target_language}"
                },
                timeout=30
            )

            response.raise_for_status()

            data = response.json()

            translated = (
                data
                .get("responseData", {})
                .get("translatedText")
            )

            return translated if translated else text

        # Long text: split into multiple requests
        chunks = split_text(text, 450)

        translated_chunks = []

        for chunk in chunks:

            response = requests.get(
                MYMEMORY_URL,
                params={
                    "q": chunk,
                    "langpair": f"{source_language}|{target_language}"
                },
                timeout=30
            )

            response.raise_for_status()

            data = response.json()

            translated = (
                data
                .get("responseData", {})
                .get("translatedText")
            )

            if translated:
                translated_chunks.append(translated)
            else:
                translated_chunks.append(chunk)

        # Combine all translated chunks
        return " ".join(translated_chunks)

    except Exception as e:

        print("Translation error:", e)

        return text