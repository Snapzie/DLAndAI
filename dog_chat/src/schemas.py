from datetime import date
from typing import Literal
from pydantic import BaseModel, Field

Award = Literal[
    "CK",
    "Bedst i Klasse",
    "Veteran CERT",
    "Veteran CACIB",
    "Nordisk Veteran CERT",
    "Reserve CACIB",
    "Nordisk CERT",
    "CERT",
    "CACIB",
    "Best of Sex",
    "Best of Breed",
    "Reserve CERT",
    "Junior CERT",
    "Junior CACIB",
    "Nordisk Junior CERT",
    "Reserve Junior CERT",
    "Junior Best of Breed",
    "Junior Best of Sex",
]

########## DOGS ##########

class DogSearchItem(BaseModel):
    dog_id: int
    pedigree_name: str
    birth_date: date | None

class DogSearchResult(BaseModel):
    dogs: list[DogSearchItem]
    has_more: bool
    error: str | None = None

class LitterDogsGroup(BaseModel):
    litter_id: int
    dogs: list[DogSearchItem]

class LitterDogsResult(BaseModel):
    litters: list[LitterDogsGroup]

class DogLitter(BaseModel):
    litter_id: int
    birth_date: date | None
    puppy_count: int | None
    sire_id: int | None
    dam_id: int | None

class DogProfile(BaseModel):
    dog_id: int
    pedigree_name: str | None
    birth_date: date | None
    sex: str | None
    from_litter: int | None
    sire_id: int | None
    dam_id: int | None
    sire_name: str | None
    dam_name: str | None
    litters: list[DogLitter] = Field(default_factory=list)

class DogProfileResult(BaseModel):
    dogs: list[DogProfile]
    missing_dog_ids: list[int]

########## TITLES ##########

class DogTitle(BaseModel):
    dog: DogSearchItem
    titles: list[str] = Field(default_factory=list)

class DogTitleResult(BaseModel):
    dogs: list[DogTitle]
    error: str | None = None

########## SHOWS ##########

class ShowItem(BaseModel):
    show_id: int
    date: date
    location: str
    organizer: str

class ShowSearchResult(BaseModel):
    shows: list[ShowItem]
    has_more: bool
    error: str | None = None

class ShowResultItem(BaseModel):
    show_id: int
    location: str
    date: date
    organizer: str
    judge: str
    dog_id: int
    pedigree_name: str
    show_class: str
    quality_grade: str
    class_placement: int | None
    sex_placement: int | None
    awards: list[str] = Field(default_factory=list)

class ShowResult(BaseModel):
    results: list[ShowResultItem]
    error: str | None = None
