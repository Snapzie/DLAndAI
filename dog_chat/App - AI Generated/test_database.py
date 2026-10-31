"""Check the database browser against isolated SQLite records."""

from datetime import date
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", execution_options={"schema_translate_map": {"dbo": None}})
        self.addCleanup(self.engine.dispose)
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("browser_database_under_test", Path(__file__).with_name("database.py"))
        self.database = importlib.util.module_from_spec(spec)
        with patch.object(sys, "path", [str(root / "src"), *sys.path]):
            with patch.dict(sys.modules, {"db": SimpleNamespace(engine=self.engine)}):
                spec.loader.exec_module(self.database)
        db = self.database
        for table in (db.dog, db.litter, db.titles, db.show, db.show_result, db.result_award):
            table.create(self.engine)
        with self.engine.begin() as connection:
            connection.execute(db.dog.insert(), [
                {"dog_id": 1, "pedigree_name": "Zulu", "sire": 2, "dam": 999, "birth_date": None, "sex": "F"},
                {"dog_id": 2, "pedigree_name": "Alpha", "sire": None, "dam": None, "birth_date": None, "sex": None},
            ])
            connection.execute(db.titles.insert(), [{"dog_id": 1, "title": "DKCH"}])
            connection.execute(db.show.insert(), [{"show_id": i, "date": date(2024, 1, i), "location": "Herning", "organizer": None} for i in (1, 2)])
            results = []
            for i, dog_id, show_id, show_class in [(1, 1, 1, "Open Class"), (2, 1, 1, "Junior Class"), (3, 1, 2, "Open Class"), (4, 2, 1, "Open Class")]:
                results.append({
                    "show_result_id": i, "dog_id": dog_id, "show_id": show_id, "class": show_class,
                    "judge": None, "quality_grade": None, "class_placement": None, "sex_placement": None,
                })
            connection.execute(db.show_result.insert(), results)
            connection.execute(db.result_award.insert(), [
                {"show_result_id": 1, "award": "CK"},
                {"show_result_id": 1, "award": "CERT"},
                {"show_result_id": 3, "award": "Junior CERT"},
                {"show_result_id": 4, "award": "CACIB"},
            ])

    def test_only_selected_dogs_are_listed_alphabetically(self):
        names = ["Tamilan Jingu-Kogo", "Stjernedrys' Known As Special Agent Z", "State-of-art Joyful Miss Norway", "Stjernedrys' Just Born To Be A Star"]
        with self.engine.begin() as connection:
            connection.execute(self.database.dog.insert(), [{"dog_id": i, "pedigree_name": name} for i, name in enumerate(names, start=3)])
        result = self.database.list_dogs()
        self.assertEqual([dog["pedigree_name"] for dog in result["dogs"]], sorted(names))
        self.assertEqual(result["total_count"], 6)
        self.assertEqual(result["show_count"], 2)
        self.assertEqual(self.database.get_dog(1)["pedigree_name"], "Zulu")

    def test_dog_details_include_parent_names_titles_and_missing_values(self):
        dog = self.database.get_dog(1)
        self.assertEqual(dog["sire_name"], "Alpha")
        self.assertEqual(dog["dam_id"], 999)
        self.assertIsNone(dog["dam_name"])
        self.assertIsNone(dog["birth_date"])
        self.assertEqual(dog["titles"], ["DKCH"])
        self.assertEqual(self.database.get_dog(2)["titles"], [])
        self.assertIsNone(self.database.get_dog(999))

    def test_shows_are_unique_per_dog_and_newest_first(self):
        self.assertEqual([show["show_id"] for show in self.database.list_shows(1)], [2, 1])
        self.assertEqual([show["show_id"] for show in self.database.list_shows(2)], [1])
        self.assertEqual(self.database.list_shows(999), [])

    def test_litters_resolve_partners_and_link_puppies_by_birth_litter(self):
        db = self.database
        with self.engine.begin() as connection:
            connection.execute(db.dog.insert(), [
                {"dog_id": 3, "pedigree_name": "Other partner", "from_litter": None, "birth_date": None},
                {"dog_id": 4, "pedigree_name": "Puppy", "from_litter": 1, "birth_date": date(2021, 1, 1)},
                {"dog_id": 5, "pedigree_name": "Unrelated puppy", "from_litter": 3, "birth_date": date(2021, 1, 1)},
            ])
            connection.execute(db.litter.insert(), [
                {"litter_id": 1, "sire": 2, "dam": 1, "birth_date": date(2021, 1, 1), "puppy_count": 3},
                {"litter_id": 2, "sire": 999, "dam": 1, "birth_date": date(2022, 1, 1), "puppy_count": None},
                {"litter_id": 3, "sire": 2, "dam": 3, "birth_date": date(2023, 1, 1), "puppy_count": 1},
            ])
        litters = db.list_litters(1)
        self.assertEqual([item["litter_id"] for item in litters], [2, 1])
        self.assertIsNone(litters[0]["partner_name"])
        self.assertEqual(litters[0]["partner_id"], 999)
        self.assertIsNone(litters[0]["puppy_count"])
        self.assertEqual(litters[0]["puppies"], [])
        self.assertEqual(litters[1]["partner_name"], "Alpha")
        self.assertEqual(litters[1]["puppy_count"], 3)
        self.assertEqual([puppy["pedigree_name"] for puppy in litters[1]["puppies"]], ["Puppy"])
        self.assertEqual([item["partner_name"] for item in db.list_litters(2)], ["Other partner", "Zulu"])
        self.assertEqual(db.list_litters(9999), [])

    def test_results_keep_classes_and_awards_separate(self):
        results = self.database.get_results(1, 1)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["show_class"], "Open Class")
        self.assertEqual(results[0]["awards"], ["CERT", "CK"])
        self.assertEqual(results[1]["show_class"], "Junior Class")
        self.assertEqual(results[1]["awards"], [])
        self.assertIsNone(results[0]["judge"])
        self.assertEqual(self.database.get_results(1, 2)[0]["awards"], ["Junior CERT"])
        self.assertEqual(self.database.get_results(2, 1)[0]["awards"], ["CACIB"])
        self.assertEqual(self.database.get_results(1, 999), [])


if __name__ == "__main__":
    unittest.main()
