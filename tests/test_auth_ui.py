"""Streamlit page smoke tests using disposable CiviScan databases."""

import tempfile
import unittest
from pathlib import Path

import auth
import database
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class AuthenticationPageTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        test_database = Path(self.temp_dir.name) / "civiscan-ui.db"
        auth.DATABASE_PATH = test_database
        database.DATABASE_PATH = test_database
        database.initialize_database()
        auth.initialize_users_table()
        self.citizen_id = auth.register_citizen(
            "UI Citizen", "ui-citizen@example.org", "+1 555 010 5555", "A-strong-ui-password-123"
        )
        self.admin_id = auth.create_initial_admin(
            name="UI Admin",
            email="ui-admin@example.org",
            employee_id="UI-ADMIN",
            department="Public Works",
            designation="Administrator",
            jurisdiction="Citywide",
            password="An-admin-ui-password-123",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _logged_in_app(self, user_id, role, name, page_key, page):
        test_app = AppTest.from_file(str(APP_PATH), default_timeout=60)
        test_app.session_state["logged_in"] = True
        test_app.session_state["authenticated_user_id"] = user_id
        test_app.session_state["user_id"] = user_id
        test_app.session_state["role"] = role
        test_app.session_state["name"] = name
        test_app.session_state[page_key] = page
        return test_app.run()

    def test_authentication_page_and_public_signup_roles(self):
        test_app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
        self.assertEqual(len(test_app.exception), 0)
        input_labels = [item.label for item in test_app.text_input]
        self.assertGreaterEqual(input_labels.count("Email or username"), 2)
        self.assertIn("Phone number", input_labels)
        self.assertFalse(any(item.label.lower() == "role" for item in test_app.selectbox))

    def test_citizen_profile_route(self):
        test_app = self._logged_in_app(
            self.citizen_id,
            auth.ROLE_CITIZEN,
            "UI Citizen",
            "navigation_Citizen",
            "My Profile",
        )
        self.assertEqual(len(test_app.exception), 0)
        self.assertTrue(any("My Profile" in str(item.value) for item in test_app.title))

    def test_citizen_dashboard_route(self):
        test_app = self._logged_in_app(
            self.citizen_id,
            auth.ROLE_CITIZEN,
            "UI Citizen",
            "navigation_Citizen",
            "Dashboard",
        )
        self.assertEqual(len(test_app.exception), 0)
        self.assertTrue(any("Citizen Dashboard" in str(item.value) for item in test_app.title))

    def test_citizen_reports_route(self):
        citizen = auth.authenticate_user("ui-citizen@example.org", "A-strong-ui-password-123")
        database.add_issue(
            str(self.citizen_id), "pothole", actor=citizen, description="UI route test"
        )
        test_app = self._logged_in_app(
            self.citizen_id,
            auth.ROLE_CITIZEN,
            "UI Citizen",
            "navigation_Citizen",
            "My Reports",
        )
        self.assertEqual(len(test_app.exception), 0)
        self.assertTrue(any("My Reports" in str(item.value) for item in test_app.title))

    def test_municipal_overview_route(self):
        test_app = self._logged_in_app(
            self.admin_id,
            auth.ROLE_MUNICIPAL_OFFICER,
            "UI Admin",
            "navigation_Municipal Officer",
            "Overview",
        )
        self.assertEqual(len(test_app.exception), 0)
        self.assertTrue(any("Overview" in str(item.value) for item in test_app.title))

    def test_incoming_media_route(self):
        test_app = self._logged_in_app(
            self.admin_id,
            auth.ROLE_MUNICIPAL_OFFICER,
            "UI Admin",
            "navigation_Municipal Officer",
            "Incoming Media",
        )
        self.assertEqual(len(test_app.exception), 0)
        self.assertTrue(any("Incoming Media" in str(item.value) for item in test_app.title))

    def test_admin_officer_management_route(self):
        test_app = self._logged_in_app(
            self.admin_id,
            auth.ROLE_MUNICIPAL_OFFICER,
            "UI Admin",
            "navigation_Municipal Officer",
            "Officer Administration",
        )
        self.assertEqual(len(test_app.exception), 0)
        self.assertTrue(any("Officer Administration" in str(item.value) for item in test_app.title))


if __name__ == "__main__":
    unittest.main()
