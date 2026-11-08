"""Which states touch which (land borders; DC counted as touching MD and VA). Used for the
"neighbouring states" clue: flu often spreads from one state into the states next to it.
Alaska, Hawaii and Puerto Rico have no land neighbours, so they get no neighbour clue."""

ADJACENT = {
    "AL": "FL GA MS TN", "AZ": "CA CO NM NV UT", "AR": "LA MO MS OK TN TX",
    "CA": "AZ NV OR", "CO": "AZ KS NE NM OK UT WY", "CT": "MA NY RI", "DE": "MD NJ PA",
    "DC": "MD VA", "FL": "AL GA", "GA": "AL FL NC SC TN", "ID": "MT NV OR UT WA WY",
    "IL": "IA IN KY MO WI", "IN": "IL KY MI OH", "IA": "IL MN MO NE SD WI",
    "KS": "CO MO NE OK", "KY": "IL IN MO OH TN VA WV", "LA": "AR MS TX", "ME": "NH",
    "MD": "DC DE PA VA WV", "MA": "CT NH NY RI VT", "MI": "IN OH WI",
    "MN": "IA ND SD WI", "MS": "AL AR LA TN", "MO": "AR IA IL KS KY NE OK TN",
    "MT": "ID ND SD WY", "NE": "CO IA KS MO SD WY", "NV": "AZ CA ID OR UT",
    "NH": "MA ME VT", "NJ": "DE NY PA", "NM": "AZ CO OK TX UT",
    "NY": "CT MA NJ PA VT", "NC": "GA SC TN VA", "ND": "MN MT SD",
    "OH": "IN KY MI PA WV", "OK": "AR CO KS MO NM TX", "OR": "CA ID NV WA",
    "PA": "DE MD NJ NY OH WV", "RI": "CT MA", "SC": "GA NC",
    "SD": "IA MN MT ND NE WY", "TN": "AL AR GA KY MO MS NC VA",
    "TX": "AR LA NM OK", "UT": "AZ CO ID NM NV WY", "VT": "MA NH NY",
    "VA": "DC KY MD NC TN WV", "WA": "ID OR", "WV": "KY MD OH PA VA",
    "WI": "IA IL MI MN", "WY": "CO ID MT NE SD UT",
}


def check_symmetric():
    """Borders go both ways: if A touches B, B must touch A. Returns any mistakes."""
    bad = []
    for a, nbs in ADJACENT.items():
        for b in nbs.split():
            if a not in ADJACENT.get(b, "").split():
                bad.append((a, b))
    return bad


def neighbor_pairs(locations):
    """locations: DataFrame with columns abbreviation, location (FIPS). Returns list of (fips, neighbour_fips)."""
    to_fips = dict(zip(locations["abbreviation"], locations["location"]))
    return [(to_fips[a], to_fips[b]) for a, nbs in ADJACENT.items() for b in nbs.split()
            if a in to_fips and b in to_fips]


if __name__ == "__main__":
    print("asymmetric borders:", check_symmetric() or "none")
