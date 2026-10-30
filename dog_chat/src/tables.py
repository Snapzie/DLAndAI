from sqlalchemy import Column, Date, Integer, MetaData, Table, Unicode
metadata = MetaData(schema="dbo")

dog = Table(
    "Dog",
    metadata,
    Column("dog_id", Integer, primary_key=True),
    Column("pedigree_name", Unicode),
    Column("from_litter", Integer),
    Column("sex", Unicode),
    Column("birth_date", Date),
    Column("sire", Integer),
    Column("dam", Integer),
)

litter = Table(
    "Litter",
    metadata,
    Column("litter_id", Integer, primary_key=True),
    Column("sire", Integer),
    Column("dam", Integer),
    Column("birth_date", Date),
    Column("puppy_count", Integer),
)

titles = Table(
    "Titles",
    metadata,
    Column("dog_id", Integer),
    Column("title", Unicode),
)

show = Table(
    "Show",
    metadata,
    Column("show_id", Integer, primary_key=True),
    Column("date", Date),
    Column("location", Unicode),
    Column("organizer", Unicode),
)

show_result = Table(
    "Show_result",
    metadata,
    Column("show_result_id", Integer),
    Column("show_id", Integer),
    Column("dog_id", Integer),
    Column("judge", Unicode),
    Column("class", Unicode),
    Column("class_placement", Integer),
    Column("quality_grade", Unicode),
    Column("sex_placement", Integer),
)

result_award = Table(
    "Result_award",
    metadata,
    Column("show_result_id", Integer),
    Column("award", Unicode)
)