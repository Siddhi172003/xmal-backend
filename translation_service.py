import requests

MYMEMORY_URL = "https://api.mymemory.translated.net/get"


def translate_text(text, target_language, source_language="en"):

    if not text:
        return text

    if target_language == "en":
        return text

    try:

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

        if translated:
            return translated

        return text

    except Exception as e:

        print("Translation error:", e)

        return text