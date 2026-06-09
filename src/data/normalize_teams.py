# Master team name mapping
# Format: "name as it appears in source" -> "our standard name"

TEAM_NAME_MAP = {
    # USA
    "United States": "USA",
    "United States of America": "USA",
    "US": "USA",
    "U.S.A.": "USA",

    # Korea
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Republic of Korea": "South Korea",

    # Iran
    "IR Iran": "Iran",
    "Islamic Republic of Iran": "Iran",

    # Ivory Coast
    "Côte d'Ivoire": "Cote d'Ivoire",
    "Ivory Coast": "Cote d'Ivoire",

    # Cape Verde
    "Cabo Verde": "Cape Verde",

    # Congo
    "Congo DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Congo": "Congo",

    # Bosnia
    "Bosnia and Herzegovina": "Bosnia",
    "Bosnia & Herzegovina": "Bosnia",

    # Czech Republic
    "Czechia": "Czech Republic",
    "Czech Republic": "Czech Republic",

    # Turkey
    "Turkiye": "Turkey",
    "Türkiye": "Turkey",

    # Colombia (common typo in some datasets)
    "Columbia": "Colombia",

    # Others
    "Curacao": "Curaçao",
    "England": "England",
    "Scotland": "Scotland",
    "Wales": "Wales",
    "Northern Ireland": "Northern Ireland",
}


def normalize_team_name(name: str) -> str:
    """
    Normalize a team name to our standard format.
    Returns the original name if no mapping exists.
    """
    if not isinstance(name, str):
        return name
    return TEAM_NAME_MAP.get(name.strip(), name.strip())


def normalize_dataframe(df, columns: list) -> object:
    """
    Apply team name normalization to specified columns in a DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of column names containing team names
    
    Returns:
        DataFrame with normalized team names
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(normalize_team_name)
    return df


if __name__ == "__main__":
    # Quick sanity check
    test_names = [
        "United States", "IR Iran", "Côte d'Ivoire",
        "Czechia", "Turkiye", "Columbia", "Korea Republic"
    ]
    print("Team name normalization test:")
    print("-" * 35)
    for name in test_names:
        print(f"  {name:30} -> {normalize_team_name(name)}")