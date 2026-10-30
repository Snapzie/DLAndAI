from datetime import date
from typing import Literal
from langchain.tools import tool
import functions
from schemas import Award, DogTitleResult

@tool
def search_dogs(pedigree_name: str) -> dict:
    """Searches the databse for a match on a whole or partial pedigree name.

    Args:
        pedigree_name: A non-empty full name or part of a name.
            Matching ignores case and searches anywhere in the name.
            Supply ordinary text, without SQL or wildcard patterns.

    Returns:
        A dictionary with:
        - dogs: Up to 20 records containing dog_id, pedigree_name,
          and birth_date. Dates use ISO format; null means unknown.
        - has_more: Whether additional matching records exist.
        - error: An explanation if the input is empty.

        An empty dogs list without an error means no matches were found.

    Use this tool to find candidate dogs by pedigree name. A partial match does
    not establish a unique identity. Ask the user to choose if multiple
    candidates match and their question requires one specific dog.
    """
    return functions.search_dogs(pedigree_name)

@tool
def get_dog_profile(dog_ids: list[int]) -> dict:
    """Retrieve profiles and recorded breeding litters for multiple dogs.

    Args:
        dog_ids: One or more dog IDs established through previous tool
            results. Never guess IDs. Batch independent lookups together.

    Returns:
        A dictionary containing:
        - dogs: Profiles for the requested dogs that were found.
        - missing_dog_ids: Requested IDs with no matching dog record.

        Each profile includes dog_id, pedigree_name, birth_date, sex,
        from_litter, sire_id, dam_id, sire_name, dam_name, and litters.
        Parent names are retrieved automatically; null means unavailable.

        from_litter identifies the litter the dog was born in.
        litters contains breeding litters where the dog is a parent.
        Each breeding litter includes litter_id, birth_date, puppy_count,
        sire_id, and dam_id.

        Dates use ISO format. Null means unknown. An empty litters list
        means no breeding litters are recorded, not proof that the dog
        has never reproduced.

    Use sire_name and dam_name directly for parent-name questions. Call this
    tool again with related dogs' IDs when further details or other unresolved
    names are needed. Multiple related dogs
    can be retrieved together. Never display internal IDs to the user.
    """
    return functions.get_dog_profile(dog_ids)

@tool
def get_litter_dogs(litter_ids: list[int]) -> dict:
    """List individually registered dogs from one or more litters.

    Args:
        litter_ids: One or more litter IDs established through previous
            tool results. Request all relevant known litters together.

    Returns:
        A dictionary containing litters, a list of groups.
        Each group includes:
        - litter_id: The requested litter ID.
        - dogs: Matching dog records containing dog_id, pedigree_name,
          and birth_date.

        Each dog belongs to the group through its from_litter reference.
        Dates use ISO format; null birth dates mean unknown.

        Every requested litter has a group, including those with no
        matching dogs. An empty dogs list means no individual dog records
        were found linked to that ID. It does not establish whether the
        litter exists or whether puppies were born.

        The number of returned dogs may differ from the litter's recorded
        puppy count.

    Use breeding-litter IDs to identify offspring. Use a dog's birth-litter
    ID to identify littermates, excluding the dog itself from those results.
    Call this tool again if additional relevant litter IDs are discovered.
    Never display internal IDs to the user.
    """
    return functions.get_litter_dogs(litter_ids)

@tool
def get_dog_titles(dog_ids: list[int]) -> dict:
    """Retrieve all recorded titles for one or more dogs.

    Args:
        dog_ids: Dog IDs established through previous tool results.
            Supply a list, even for one dog. Never guess IDs.

    Returns:
        A dictionary containing:
        - dogs: One entry per matching dog, with:
          - dog: dog_id, pedigree_name, and birth_date.
          - titles: A list of recorded title strings.
        - error: An explanation if the input list is empty; otherwise null.

        Dogs without recorded titles have an empty titles list.
        Requested dogs that were not found are omitted from dogs.

    Batch independent lookups together. Preserve titles as recorded;
    do not invent meanings for unfamiliar abbreviations. An empty titles
    list means no titles are recorded, not proof none were ever earned.
    Never display internal IDs to the user.
    """
    return functions.get_dog_titles(dog_ids)


TITLE_NAMES = {
    "DANISH_CHAMPION": "Danish champion; Dansk champion",
    "SECH": "Swedish champion; Svensk champion",
    "CIB": "International champion; international champion",
    "DKJUCH": "Danish kunior champion; dansk junior champion",
    "KLBJCH": "Danish club junior champion; dansk klub junior champion",
}
TITLE_DESCRIPTION = "\n".join(f"- {code}: {names}" for code, names in TITLE_NAMES.items())
@tool
def get_dogs_by_titles(titles: list[str]) -> dict:
    """Find dogs with any of the requested recorded titles.

    Args:
        titles: A list of canonical title codes, even for a single title.

    Returns:
        A dictionary with dogs and an optional error. Each entry contains:
        - dog: dog_id, pedigree_name, and birth_date.
        - titles: Only the requested titles that this dog matched.
        Each dog matches at least one requested title.
        An empty dogs list without an error means no matches were found.

    Use the returned pedigree names directly. Use get_dog_titles when all
    recorded titles for a dog are requested.
    Never display internal IDs. Missing records do not prove no title was earned.
    """
    titles = list(dict.fromkeys(title.strip().upper() for title in titles))
    if not titles or any(title not in TITLE_NAMES for title in titles):
        return DogTitleResult(dogs=[],error="Provide supported title codes.")

    return functions.get_dogs_by_titles(titles)

get_dogs_by_titles.description += (
    "\n\nSupported titles:\n"
    + TITLE_DESCRIPTION
    + "\n\nResolve the user's wording to a supported abbreviation before calling."
    + "\nIf the intended title is ambiguous, ask for clarification."
)


@tool
def search_show(show_location: str | None = None, show_interval_start: date | None = None, show_interval_end: date | None = None) -> dict:
    """Find shows by location, date range, or both. Supply at least one filter.

    Args:
        show_location: Optional full or partial location text. Matching ignores
            case and searches anywhere in the location. Do not supply SQL or
            wildcards. Omit when searching by dates alone.
        show_interval_start: Optional inclusive start date in YYYY-MM-DD format.
        show_interval_end: Optional inclusive end date in YYYY-MM-DD format.
            Either boundary may be omitted. For a whole year such as 2024,
            use 2024-01-01 through 2024-12-31. For a single day, use the same
            date for both boundaries. Do not invent dates the user did not specify.

    Returns:
        A dictionary containing:
        - shows: Up to 20 records with show_id, location, date, and organizer.
          Exact location matches appear first, then shows are ordered by date.
        - has_more: Whether additional matching shows were omitted.
        - error: An explanation if neither a location nor a date boundary is supplied.

        Dates use ISO format. Null means unknown.
        An empty shows list without an error means no matching records.

    Multiple shows can share a location and date range. When a question requires
    one show and several match, ask for clarification using dates, locations,
    and organizers. Do not choose a show arbitrarily or assume an exact
    location match identifies a unique show.

    If has_more is true, explain that results are incomplete and ask for
    a more specific location or date range when needed.
    Never display internal show IDs to the user.
    """
    return functions.search_show(show_location, show_interval_start, show_interval_end)

@tool
def get_show_results(show_ids: list[int] | None = None, dog_ids: list[int] | None = None, show_interval_start: date | None = None, show_interval_end: date | None = None, sex_placements: list[Literal[1, 2, 3, 4, "unplaced"]] | None = None, awards: list[Award] | None = None) -> dict:
    """Retrieve recorded show results for shows, dogs, or both.

    Args:
        show_ids: Optional list of show IDs established through previous tool
            results. Never guess IDs.
        dog_ids: Optional list of dog IDs established through previous tool
            results. Supply a list even for one dog.
        show_interval_start: Optional inclusive start date in YYYY-MM-DD format.
        show_interval_end: Optional inclusive end date in YYYY-MM-DD format.
            For a whole year, use January 1 through December 31.
        sex_placements: Optional list matching any supplied sex placement:
            1, 2, 3, 4, or "unplaced". "Not placed" and "Ikke placeret" mean
            outside the top four and map to "unplaced". Use [1] for first place,
            [1, 2] for the top two, and ["unplaced"] for outside the top four.
            None or an empty list means no sex-placement filter.
        awards: Optional list of normalized award labels. Results match any
            supplied award; None or an empty list means no award filter.
            Resolve abbreviations to these labels:
            - CK: CK.
            - BIK: Bedst i Klasse.
            - Vet.CERT: Veteran CERT.
            - Vet.CACIB: Veteran CACIB.
            - N-VCAC: Nordisk Veteran CERT.
            - R.CACIB: Reserve CACIB.
            - NCAC: Nordisk CERT.
            - CERT: CERT.
            - CACIB: CACIB.
            - BIM: Best of Sex.
            - BIR / BOB / Best of Breed: Best of Breed.
            - R.CERT: Reserve CERT.
            - Jun.CERT / J.CERT: Junior CERT.
            - Jun.CACIB / J.CACIB: Junior CACIB.
            - N-JCAC: Nordisk Junior CERT.
            - R.Jun.CERT / R.J.CERT: Reserve Junior CERT.
            - BIR Junior / BIR Jun.: Junior Best of Breed.
            - BIM Junior / BIM Jun.: Junior Best of Sex.
            Keep junior, veteran, and reserve awards distinct. Ask for
            clarification when the intended award is ambiguous.

    Supply at least one non-empty show_ids or dog_ids list; dates alone are
    insufficient. Supplied filters combine with AND: when both ID lists are
    provided, results must match a listed show and a listed dog.

    Returns:
        A dictionary with results and an optional error. Each result contains
        show_id, location, date, organizer, judge, dog_id, pedigree_name,
        show_class, quality_grade, class_placement, sex_placement, and awards.
        awards is a list of all recorded award strings for that specific result,
        including awards other than those used to filter.
        An empty awards list means no awards are recorded for that result,
        not proof that none were earned. Results without awards are included
        when no award filter is supplied.
        Dates use ISO format. Only the top four dogs receive placements.
        A null class_placement or sex_placement means the dog did not place
        in the top four of that competition. Quality grades are separate.
        An empty results list without an error means no matching records.
        An error is returned if neither ID list contains IDs.

    Use returned pedigree names and show details directly. Preserve recorded
    classes, grades, and awards; do not guess unfamiliar codes or infer awards
    from grades or placements. Keep awards attached to their specific result,
    including when the same dog has multiple results.
    Missing records do not prove a dog never participated. Never display IDs.
    """
    return functions.get_show_results(show_ids, dog_ids, show_interval_start, show_interval_end, sex_placements, awards)
