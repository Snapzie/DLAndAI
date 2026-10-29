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