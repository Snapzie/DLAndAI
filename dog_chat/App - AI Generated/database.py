"""Read-only queries for the database browser, independent of chat memory."""

from sqlalchemy import case, func, or_, select

from db import engine
from tables import dog, litter, titles, show, show_result, result_award


def list_dogs():
    selected_names = (
        "Tamilan Jingu-Kogo",
        "Stjernedrys' Known As Special Agent Z",
        "State-of-art Joyful Miss Norway",
        "Stjernedrys' Just Born To Be A Star",
    )
    statement = (
        select(dog.c.dog_id, dog.c.pedigree_name)
        .where(dog.c.pedigree_name.in_(selected_names))
        .order_by(dog.c.pedigree_name, dog.c.dog_id)
    )
    with engine.connect() as connection:
        dogs = [dict(row) for row in connection.execute(statement).mappings()]
        total_count = connection.scalar(select(func.count()).select_from(dog))
        show_count = connection.scalar(select(func.count()).select_from(show))
    return {"dogs": dogs, "total_count": total_count, "show_count": show_count}


def get_dog(dog_id):
    sire = dog.alias("sire")
    dam = dog.alias("dam")
    statement = (
        select(
            dog.c.dog_id, dog.c.pedigree_name, dog.c.birth_date, dog.c.sex,
            dog.c.sire.label("sire_id"), dog.c.dam.label("dam_id"),
            sire.c.pedigree_name.label("sire_name"), dam.c.pedigree_name.label("dam_name"),
        )
        .select_from(dog.outerjoin(sire, dog.c.sire == sire.c.dog_id).outerjoin(dam, dog.c.dam == dam.c.dog_id))
        .where(dog.c.dog_id == dog_id)
    )
    with engine.connect() as connection:
        row = connection.execute(statement).mappings().first()
        if row is None:
            return None
        result = dict(row)
        result["titles"] = connection.execute(
            select(titles.c.title).where(titles.c.dog_id == dog_id).order_by(titles.c.title)
        ).scalars().all()
    return result


def list_shows(dog_id):
    statement = (
        select(show.c.show_id, show.c.date, show.c.location, show.c.organizer)
        .join(show_result, show.c.show_id == show_result.c.show_id)
        .where(show_result.c.dog_id == dog_id)
        .distinct()
        .order_by(show.c.date.desc(), show.c.show_id)
    )
    with engine.connect() as connection:
        return [dict(row) for row in connection.execute(statement).mappings()]


def list_litters(dog_id):
    sire = dog.alias("sire")
    dam = dog.alias("dam")
    statement = (
        select(
            litter.c.litter_id, litter.c.birth_date, litter.c.puppy_count,
            case((litter.c.sire == dog_id, litter.c.dam), else_=litter.c.sire).label("partner_id"),
            case((litter.c.sire == dog_id, dam.c.pedigree_name), else_=sire.c.pedigree_name).label("partner_name"),
        )
        .select_from(litter.outerjoin(sire, litter.c.sire == sire.c.dog_id).outerjoin(dam, litter.c.dam == dam.c.dog_id))
        .where(or_(litter.c.sire == dog_id, litter.c.dam == dog_id))
        .order_by(litter.c.birth_date.desc(), litter.c.litter_id)
    )
    with engine.connect() as connection:
        litters = {row["litter_id"]: dict(row, puppies=[]) for row in connection.execute(statement).mappings()}
        if litters:
            puppies = connection.execute(
                select(dog.c.from_litter, dog.c.pedigree_name, dog.c.birth_date)
                .where(dog.c.from_litter.in_(litters))
                .order_by(dog.c.pedigree_name, dog.c.dog_id)
            ).mappings().all()
            for puppy in puppies:
                litters[puppy["from_litter"]]["puppies"].append({"pedigree_name": puppy["pedigree_name"], "birth_date": puppy["birth_date"]})
    return list(litters.values())


def get_results(dog_id, show_id):
    statement = (
        select(
            show_result.c.show_result_id, show_result.c["class"].label("show_class"),
            show_result.c.judge, show_result.c.quality_grade,
            show_result.c.class_placement, show_result.c.sex_placement, result_award.c.award,
        )
        .outerjoin(result_award, show_result.c.show_result_id == result_award.c.show_result_id)
        .where(show_result.c.dog_id == dog_id, show_result.c.show_id == show_id)
        .order_by(show_result.c.show_result_id, result_award.c.award)
    )
    with engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()
    results = {}
    for row in rows:
        result_id = row["show_result_id"]
        if result_id not in results:
            results[result_id] = {key: value for key, value in row.items() if key not in {"show_result_id", "award"}}
            results[result_id]["awards"] = []
        if row["award"] is not None:
            results[result_id]["awards"].append(row["award"])
    return list(results.values())
