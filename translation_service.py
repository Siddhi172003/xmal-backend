import requests


MYMEMORY_URL = "https://api.mymemory.translated.net/get"


def translate_text(text, target_language, source_language="en"):

    if not text:
        return text

    # No translation needed for English
    if not target_language or target_language == "en":
        return text

    try:

        response = requests.get(
            MYMEMORY_URL,
            params={
                "q": text,
                "langpair": f"{source_language}|{target_language}"
            },
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        translated_text = (
            data
            .get("responseData", {})
            .get("translatedText")
        )

        if translated_text:
            return translated_text

        print("MyMemory did not return a translation.")
        return text

    except Exception as e:

        print(
            "MyMemory translation error:",
            str(e)
        )

        # Keep original SHAP explanation
        # if translation fails.
        return text