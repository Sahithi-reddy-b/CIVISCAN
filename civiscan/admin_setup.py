"""One-time interactive bootstrap for the first municipal administrator."""

from getpass import getpass

from auth import create_initial_admin, has_active_admin, initialize_users_table


def main():
    initialize_users_table()
    if has_active_admin():
        raise SystemExit("An active CiviScan administrator is already configured.")

    print("Create the initial Municipal Officer administrator account.")
    name = input("Full name: ").strip()
    email = input("Official email: ").strip()
    employee_id = input("Employee ID: ").strip()
    department = input("Department: ").strip()
    designation = input("Designation: ").strip()
    jurisdiction = input("Assigned jurisdiction: ").strip()
    password = getpass("Password (minimum 12 characters): ")
    confirmation = getpass("Confirm password: ")
    if password != confirmation:
        raise SystemExit("Passwords did not match; no account was created.")

    try:
        user_id = create_initial_admin(
            name=name,
            email=email,
            employee_id=employee_id,
            department=department,
            designation=designation,
            jurisdiction=jurisdiction,
            password=password,
        )
    except (ValueError, PermissionError) as error:
        raise SystemExit(str(error)) from error
    print(f"Initial administrator created (user ID {user_id}).")


if __name__ == "__main__":
    main()