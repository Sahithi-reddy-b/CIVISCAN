"""Regression tests for the additive CiviScan authentication migration."""

import hashlib
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

import auth
import database
from civiscan.security import AuthorizationError, require_role


class AuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "legacy.db"
        auth.DATABASE_PATH = self.database_path
        database.DATABASE_PATH = self.database_path
        self._create_legacy_schema()
        auth.initialize_users_table()
        auth.initialize_users_table()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_legacy_schema(self):
        salt = b"civiscan-test-salt"
        password = "LegacyPassword!234"
        password_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000).hex()
        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute(
                "CREATE TABLE users (user_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, "
                "username TEXT NOT NULL UNIQUE, password TEXT NOT NULL, "
                "role TEXT NOT NULL CHECK (role IN ('Citizen', 'Municipal Officer')))"
            )
            connection.execute(
                "INSERT INTO users(user_id, name, username, password, role) VALUES (7, ?, ?, ?, ?)",
                ("Legacy Citizen", "legacy-citizen", f"{salt.hex()}${password_hash}", auth.ROLE_CITIZEN),
            )
            connection.execute(
                "INSERT INTO users(user_id, name, username, password, role) VALUES (9, ?, ?, ?, ?)",
                ("Legacy Officer", "legacy-officer", f"{salt.hex()}${password_hash}", auth.ROLE_MUNICIPAL_OFFICER),
            )
            connection.execute(database.ISSUES_TABLE_SQL)
            connection.execute(
                "INSERT INTO issues(issue_id, user_id, category) VALUES (41, '7', 'pothole')"
            )
            connection.execute(
                "INSERT INTO issues(issue_id, user_id, category) VALUES (42, '9', 'pothole')"
            )
            connection.commit()
        finally:
            connection.close()

    def test_migration_is_additive_and_idempotent(self):
        connection = sqlite3.connect(self.database_path)
        try:
            users = connection.execute("SELECT user_id, username FROM users ORDER BY user_id").fetchall()
            issue = connection.execute("SELECT issue_id, user_id FROM issues").fetchone()
            columns = {row[1] for row in connection.execute("PRAGMA table_info(users)")}
        finally:
            connection.close()
        self.assertEqual(users, [(7, "legacy-citizen"), (9, "legacy-officer")])
        self.assertEqual(issue, (41, "7"))
        self.assertTrue({"email", "phone_number", "password_hash", "employee_id", "department", "designation", "jurisdiction", "account_status", "is_admin"}.issubset(columns))

    def test_legacy_login_rehashes_to_argon2_and_preserves_id(self):
        user = auth.authenticate_user("legacy-citizen", "LegacyPassword!234", auth.ROLE_CITIZEN)
        self.assertEqual(user["user_id"], 7)
        connection = sqlite3.connect(self.database_path)
        try:
            password_hash = connection.execute("SELECT password_hash FROM users WHERE user_id = 7").fetchone()[0]
        finally:
            connection.close()
        self.assertTrue(password_hash.startswith("$argon2id$"))
        self.assertIsNone(auth.authenticate_user("legacy-citizen", "wrong-password", auth.ROLE_CITIZEN))
        self.assertIsNone(auth.authenticate_user("legacy-citizen", "LegacyPassword!234", auth.ROLE_MUNICIPAL_OFFICER))

    def test_registration_duplicate_email_and_public_officer_signup(self):
        user_id = auth.register_citizen("New Citizen", "new@example.org", "+1 555 010 1234", "A-strong-password-123")
        self.assertEqual(auth.authenticate_user("NEW@example.org", "A-strong-password-123")["user_id"], user_id)
        with self.assertRaises(ValueError):
            auth.register_citizen("Another Citizen", "NEW@example.org", "+1 555 010 9999", "A-strong-password-456")
        with self.assertRaises(ValueError):
            auth.register_user("Public Officer", "officer@example.org", "A-strong-password-456", auth.ROLE_MUNICIPAL_OFFICER)

    def test_issue_database_operations_check_role_and_ownership(self):
        citizen = auth.authenticate_user("legacy-citizen", "LegacyPassword!234")
        officer = auth.authenticate_user("legacy-officer", "LegacyPassword!234")
        new_issue_id = database.add_issue("7", "pothole", actor=citizen)
        self.assertEqual(database.get_issue_by_id(new_issue_id, actor=citizen)["user_id"], "7")
        self.assertEqual(len(database.get_issues_by_user("7", actor=citizen)), 2)
        with self.assertRaises(AuthorizationError):
            database.add_issue("7", "pothole", actor=officer)
        with self.assertRaises(AuthorizationError):
            database.get_issue_by_id(42, actor=citizen)
        with self.assertRaises(AuthorizationError):
            database.get_issues_by_user("9", actor=citizen)
        with self.assertRaises(AuthorizationError):
            database.get_all_issues(actor=citizen)
        with self.assertRaises(AuthorizationError):
            database.update_issue_status(new_issue_id, "RESOLVED", actor=citizen)
        self.assertTrue(database.update_issue_status(new_issue_id, "VERIFIED", actor=officer))
        self.assertTrue(database.update_issue_priority(new_issue_id, 65.0, "Medium", actor=officer))

    def test_profile_owner_and_password_change(self):
        citizen = auth.authenticate_user("legacy-citizen", "LegacyPassword!234")
        with self.assertRaises(PermissionError):
            auth.get_user_by_id(9, actor=citizen)
        with self.assertRaises(ValueError):
            auth.change_password(citizen, "incorrect-current", "A-new-password-12345")
        auth.update_user_profile(
            citizen,
            7,
            full_name="Updated Legacy Citizen",
            email="legacy@example.org",
            phone_number="+1 555 010 4444",
        )
        updated = auth.get_user_by_id(7, actor=citizen)
        self.assertEqual(updated["email"], "legacy@example.org")
        auth.change_password(citizen, "LegacyPassword!234", "A-new-password-12345")
        self.assertIsNone(auth.authenticate_user("legacy@example.org", "LegacyPassword!234"))
        self.assertEqual(auth.authenticate_user("legacy@example.org", "A-new-password-12345")["user_id"], 7)

    def test_admin_only_officer_creation_and_deactivation(self):
        citizen = auth.authenticate_user("legacy-citizen", "LegacyPassword!234")
        with self.assertRaises(PermissionError):
            auth.create_officer(
                citizen,
                name="Restricted Officer", email="restricted@example.org", employee_id="E-1",
                department="Roads", designation="Inspector", jurisdiction="North",
                password="A-strong-password-456",
            )

        admin_id = auth.create_initial_admin(
            name="Initial Admin", email="admin@example.org", employee_id="ADM-1",
            department="Public Works", designation="Director", jurisdiction="Citywide",
            password="An-admin-password-123",
        )
        admin = auth.authenticate_user("admin@example.org", "An-admin-password-123", auth.ROLE_MUNICIPAL_OFFICER)
        officer_id = auth.create_officer(
            admin,
            name="Officer One", email="officer@example.org", employee_id="OFF-1",
            department="Roads", designation="Engineer", jurisdiction="North",
            password="An-officer-password-123",
        )
        with self.assertRaises(PermissionError):
            auth.create_initial_admin(
                name="Duplicate Admin", email="other-admin@example.org", employee_id="ADM-2",
                department="Public Works", designation="Director", jurisdiction="Citywide",
                password="Another-admin-password-123",
            )
        officer = auth.authenticate_user("officer@example.org", "An-officer-password-123")
        officer_session = {}
        auth.establish_session(officer_session, officer)
        auth.update_officer(admin, officer_id, account_status="inactive")
        self.assertIsNone(auth.get_current_user(officer_session))
        self.assertNotIn("authenticated_user_id", officer_session)
        self.assertIsNone(auth.authenticate_user("officer@example.org", "An-officer-password-123"))
        with self.assertRaises(ValueError):
            auth.update_officer(admin, admin_id, account_status="inactive", is_admin=False)

    def test_lockout_role_guard_session_and_logout(self):
        auth.register_citizen("Lockout User", "lockout@example.org", "+1 555 010 2222", "A-lockout-password-123")
        for _ in range(auth.AUTH_MAX_LOGIN_ATTEMPTS):
            self.assertIsNone(auth.authenticate_user("lockout@example.org", "incorrect-password"))
        self.assertIsNone(auth.authenticate_user("lockout@example.org", "A-lockout-password-123"))
        citizen = auth.authenticate_user("legacy-citizen", "LegacyPassword!234")
        with self.assertRaises(AuthorizationError):
            require_role(citizen, {auth.ROLE_MUNICIPAL_OFFICER})

        user = auth.authenticate_user("legacy-citizen", "LegacyPassword!234")
        state = {}
        auth.establish_session(state, user)
        self.assertEqual(auth.get_current_user(state)["user_id"], 7)
        auth.clear_session(state)
        self.assertIsNone(auth.get_current_user(state))

    def test_one_time_expiring_reset_token(self):
        email = "reset@example.org"
        user_id = auth.register_citizen("Reset User", email, "+1 555 010 3333", "A-reset-password-123")
        token = "temporary-token-for-test"
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute(
                "INSERT INTO password_reset_tokens(user_id, token_hash, expires_at) VALUES (?, ?, ?)",
                (user_id, token_hash, int(time.time()) + 600),
            )
            connection.commit()
        finally:
            connection.close()
        self.assertTrue(auth.reset_password(token, "A-new-reset-password-123"))
        self.assertFalse(auth.reset_password(token, "A-reused-reset-password-123"))
        self.assertEqual(auth.authenticate_user(email, "A-new-reset-password-123")["user_id"], user_id)


if __name__ == "__main__":
    unittest.main()
