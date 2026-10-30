from sqlalchemy import case, func, or_, select
from typing import Literal
from datetime import date
from db import engine
from schemas import Award
from tables import dog, litter, titles, show, show_result, result_award
from schemas import DogSearchItem, DogSearchResult, DogLitter, DogProfile, DogProfileResult, LitterDogsResult, LitterDogsGroup, DogTitle, DogTitleResult, ShowItem, ShowSearchResult, ShowResultItem, ShowResult


def search_dogs(pedigree_name: str) -> dict:
    name = pedigree_name.strip()
    search_limit = 20

    if not name:
        return DogSearchResult(
            dogs = [],
            has_more = False,
            error = "Provide a pedigree name."
        ).model_dump(mode='json')

    statement = (
        select(
            dog.c.dog_id,
            dog.c.pedigree_name,
            dog.c.birth_date,
        )
        .where(
            func.charindex(
                func.lower(name),
                func.lower(dog.c.pedigree_name)
            ) > 0
        )
        .order_by(
            case((func.lower(dog.c.pedigree_name) == func.lower(name), 0), else_=1), # Exact naming matches first
            dog.c.pedigree_name,
            dog.c.dog_id,
        )
        .limit(search_limit + 1)
    )

    with engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()

    result = DogSearchResult(
        dogs = [DogSearchItem.model_validate(row) for row in rows[:search_limit]],
        has_more = len(rows) > search_limit
    )
    return result.model_dump(mode='json',exclude_none=True)

def get_dog_profile(dog_ids: list[int]) -> dict:
    # Remove duplicates while preserving order.
    dog_ids = list(dict.fromkeys(dog_ids))

    if not dog_ids:
        return DogProfileResult(
            dogs=[],
            missing_dog_ids=[],
        ).model_dump(mode="json")

    sire = dog.alias("sire")
    dam = dog.alias("dam")
    profile_statement = (
        select(
            dog.c.dog_id,
            dog.c.pedigree_name,
            dog.c.birth_date,
            dog.c.sex,
            dog.c.from_litter,
            dog.c.sire.label("sire_id"),
            dog.c.dam.label("dam_id"),
            sire.c.pedigree_name.label("sire_name"),
            dam.c.pedigree_name.label("dam_name"),
        )
        .select_from(
            dog.outerjoin(sire, dog.c.sire == sire.c.dog_id)
            .outerjoin(dam, dog.c.dam == dam.c.dog_id)
        )
        .where(dog.c.dog_id.in_(dog_ids))
    )

    litters_statement = (
        select(
            litter.c.litter_id,
            litter.c.birth_date,
            litter.c.puppy_count,
            litter.c.sire.label("sire_id"),
            litter.c.dam.label("dam_id"),
        )
        .where(
            or_(
                litter.c.sire.in_(dog_ids),
                litter.c.dam.in_(dog_ids),
            )
        )
        .order_by(
            litter.c.birth_date.desc(),
            litter.c.litter_id.desc(),
        )
    )

    with engine.connect() as connection:
        profile_rows = (
            connection.execute(profile_statement).mappings().all()
        )

        litter_rows = (
            connection.execute(litters_statement).mappings().all()
        )

    profiles = {row["dog_id"]: DogProfile.model_validate(row) for row in profile_rows}
    # Add litters to each dog in profiles
    for row in litter_rows:
        breeding_litter = DogLitter.model_validate(row)
        for parent_id in {breeding_litter.sire_id, breeding_litter.dam_id}:
            if parent_id in profiles:
                profiles[parent_id].litters.append(breeding_litter)

    result = DogProfileResult(
        dogs = [profiles[dog_id] for dog_id in dog_ids if dog_id in profiles],
        missing_dog_ids = [dog_id for dog_id in dog_ids if dog_id not in profiles]
    )
    return result.model_dump(mode="json")

def get_litter_dogs(litter_ids: list[int]) -> dict:
    litter_ids = list(dict.fromkeys(litter_ids))

    if not litter_ids:
        return LitterDogsResult(
            litters=[]
        ).model_dump(mode="json")

    statement = (
        select(
            dog.c.dog_id,
            dog.c.pedigree_name,
            dog.c.birth_date,
            dog.c.from_litter
        )
        .where(dog.c.from_litter.in_(litter_ids))
        .order_by(dog.c.pedigree_name, dog.c.dog_id)
    )

    with engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()

    groups = {litter_id: LitterDogsGroup(litter_id=litter_id,dogs=[]) for litter_id in litter_ids}
    for row in rows:
        groups[row["from_litter"]].dogs.append(DogSearchItem.model_validate(row))

    result = LitterDogsResult(litters=list(groups.values()))
    return result.model_dump(mode="json")

def get_dog_titles(dog_ids: list[int]) -> dict:
    # Remove duplicates while preserving order.
    dog_ids = list(dict.fromkeys(dog_ids))

    if not dog_ids:
        return DogTitleResult(
            dogs=[],
            error = "Provide dog IDs for the dogs to look up."
        ).model_dump(mode="json")

    statement = (
        select(
            dog.c.dog_id,
            dog.c.pedigree_name,
            dog.c.birth_date,
            titles.c.title
        )
        .outerjoin(titles, dog.c.dog_id == titles.c.dog_id)
        .where(dog.c.dog_id.in_(dog_ids))
    )

    with engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()

    groupings = {}
    for row in rows:
        dog_id = row["dog_id"]
        if dog_id not in groupings:
            groupings[dog_id] = DogTitle(dog=DogSearchItem.model_validate(row))
        if row["title"] is not None:
            groupings[dog_id].titles.append(row["title"])
    result = DogTitleResult(dogs=list(groupings.values()))
    return result.model_dump(mode="json")

def get_dogs_by_titles(titles_to_search: list[str]) -> dict:
    # Remove duplicates while preserving order.
    titles_to_search = list(dict.fromkeys(titles_to_search))

    if not titles_to_search:
        return DogTitleResult(dogs=[],error = "Provide titles to search for.").model_dump(mode="json")

    statement = (
        select(
            dog.c.dog_id,
            dog.c.pedigree_name,
            dog.c.birth_date,
            titles.c.title
        )
        .join(titles,dog.c.dog_id == titles.c.dog_id)
        .where(titles.c.title.in_(titles_to_search))
    )

    with engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()

    groupings = {}
    for row in rows:
        dog_id = row["dog_id"]
        if dog_id not in groupings:
            groupings[dog_id] = DogTitle(dog=DogSearchItem.model_validate(row))
        if row["title"] is not None:
            groupings[dog_id].titles.append(row["title"])
    result = DogTitleResult(dogs=list(groupings.values()))
    return result.model_dump(mode="json")

def search_show(show_location: str | None = None, show_interval_start: date | None = None, show_interval_end: date | None = None) -> dict:
    show_location = (show_location or "").strip()
    search_limit = 20

    if not show_location and not show_interval_start and not show_interval_end:
        return ShowSearchResult(
            shows = [],
            has_more = False,
            error = "Provide a more narrow search by supplying a location or date interval."
        ).model_dump(mode='json')

    statement = (
        select(
            show.c.show_id,
            show.c.location,
            show.c.date,
            show.c.organizer
        )
        .order_by(
            case((func.lower(show.c.location) == func.lower(show_location), 0), else_=1), # Exact naming matches first
            show.c.date,
            show.c.show_id
        )
        .limit(search_limit+1)
    )

    if show_location:
        statement = statement.where(
            func.charindex(
                func.lower(show_location),
                func.lower(show.c.location)
            ) > 0
        )

    if show_interval_start is not None:
        statement = statement.where(
            show.c.date >= show_interval_start,
        )
    if show_interval_end is not None:
        statement = statement.where(
            show.c.date <= show_interval_end
        )

    with engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()

    result = ShowSearchResult(
        shows = [ShowItem.model_validate(row) for row in rows[:search_limit]],
        has_more = len(rows) > search_limit
    )
    return result.model_dump(mode='json',exclude_none=True)

def get_show_results(
        show_ids: list[int] | None = None, 
        dog_ids: list[int] | None = None, 
        show_interval_start: date | None = None, 
        show_interval_end: date | None = None,
        sex_placements: list[Literal[1, 2, 3, 4, "unplaced"]] | None = None,
        awards: list[Award] | None = None
    ) -> dict:

    if not show_ids and not dog_ids:
        return ShowResult(
            results = [],
            error = "Provide a more narrow search by supplying show IDs or Dog IDs."
        ).model_dump(mode='json')

    statement = (
        select(
            show_result.c.show_result_id,
            show.c.show_id,
            show.c.location,
            show.c.date,
            show.c.organizer,
            show_result.c.judge,
            dog.c.dog_id,
            dog.c.pedigree_name,
            show_result.c["class"].label("show_class"),
            show_result.c.quality_grade,
            show_result.c.class_placement,
            show_result.c.sex_placement,
            result_award.c.award
        )
        .select_from(dog)
        .join(show_result,dog.c.dog_id == show_result.c.dog_id)
        .join(show,show_result.c.show_id == show.c.show_id)
        .outerjoin(result_award,result_award.c.show_result_id == show_result.c.show_result_id)
    )

    if show_ids:
        statement = statement.where(
            show.c.show_id.in_(show_ids)
        )
    if dog_ids:
        statement = statement.where(
            dog.c.dog_id.in_(dog_ids)
        )
    if show_interval_start is not None:
        statement = statement.where(
            show.c.date >= show_interval_start
        )
    if show_interval_end is not None:
        statement = statement.where(
            show.c.date <= show_interval_end
        )
    if sex_placements:
        placements = [value for value in sex_placements if value != "unplaced"]
        condition = show_result.c.sex_placement.in_(placements)
        if "unplaced" in sex_placements:
            condition = or_(condition, show_result.c.sex_placement.is_(None))
        statement = statement.where(condition)

    if awards:
        statement = statement.where(
            show_result.c.show_result_id.in_(
                select(result_award.c.show_result_id)
                .where(result_award.c.award.in_(awards))
            )
        )

    with engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()

    groupings = {}
    for row in rows:
        show_result_id = row["show_result_id"]
        if show_result_id not in groupings:
            groupings[show_result_id] = ShowResultItem.model_validate(row)
        if row["award"] is not None:
            groupings[show_result_id].awards.append(row["award"])
    return ShowResult(results=list(groupings.values())).model_dump(mode="json")
