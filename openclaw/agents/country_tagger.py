class CountryTagger:
    def __init__(self, entities: dict):
        self.entities = entities or {}
    def tag(self, title: str, summary: str) -> str:
        text = f"{title} {summary}".lower()
        for code, keys in self.entities.items():
            if code == "GLOBAL":
                continue
            for k in keys or []:
                if str(k).lower() in text:
                    return code
        return "GLOBAL"
