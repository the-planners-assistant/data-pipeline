def classify_chunk_type(path: str) -> str:
    path = path or ""

    if "Glossary/Definitions" in path:
        return "GlossaryDefinition"
    elif "Glossary/Abbreviations" in path:
        return "GlossaryAbbreviation"
    elif "Policy_" in path:
        return "PolicyStatement"
    elif "SupportingText_" in path:
        return "SupportingText"
    elif "Annex" in path:
        return "Appendix"
    elif "Table" in path:
        return "Table"
    elif "Figure" in path:
        return "Figure"
    elif any(key in path for key in ["Foreword", "Introduction", "Purpose", "Scope"]):
        return "IntroSection"
    elif path == "Unknown":
        return "Unmapped"
    else:
        return "Other"
