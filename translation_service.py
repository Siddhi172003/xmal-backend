import os

from google.cloud import translate_v3


def translate_text(
        text,
        target_language):

    # English doesn't need translation

    if not target_language or \
            target_language == "en":

        return text


    project_id = os.environ.get(
        "GOOGLE_CLOUD_PROJECT"
    )


    if not project_id:

        print(
            "GOOGLE_CLOUD_PROJECT is missing"
        )

        return text


    client = (
        translate_v3
        .TranslationServiceClient()
    )


    parent = (
        f"projects/{project_id}"
        f"/locations/global"
    )


    try:

        response = client.translate_text(

            request={

                "parent":
                    parent,

                "contents":
                    [text],

                "mime_type":
                    "text/plain",

                "source_language_code":
                    "en",

                "target_language_code":
                    target_language

            }
        )


        if not response.translations:

            return text


        return (
            response
            .translations[0]
            .translated_text
        )


    except Exception as e:

        print(
            "Translation error:",
            str(e)
        )

        # Don't break malware scanning
        # if translation fails.

        return text