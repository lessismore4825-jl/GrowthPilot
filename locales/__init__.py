from .en import TEXT as EN_TEXT
from .zh import TEXT as ZH_TEXT


LANGUAGE_MAP = {

    "English":
        EN_TEXT,

    "中文":
        ZH_TEXT,

}


def get_text(language="English"):

    """
    Return selected language dictionary accessor.

    Example:

    t = get_text("English")

    t("app_title")

    """

    selected_language = LANGUAGE_MAP.get(
        language,
        EN_TEXT,
    )


    def translate(key):

        return selected_language.get(
            key,
            key,
        )


    return translate