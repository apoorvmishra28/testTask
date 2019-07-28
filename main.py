from .db import DB

def get_admins_option():
    print("Welcome {}".format(user[1]))
    print("*****Use given options to perform actions*****")
    print("1) List all products.")
    print("2) Add products.")
    print("3) Delete products.")
    print("4) List order report.")
    choice = int(input("Enter choice: "))
    return choice

def get_users_option():
    print("Welcome {}".format(user[1]))
    print("*****Use given options to perform actions*****")
    print("1) List all products.")
    print("2) Buy product.")
    print("3) Book order.")
    print("4) Get Order History.")
    choice = int(input("Enter choice: "))
    return choice

def get_item_name(db, item_id):
    name = db.get_items_name(item_id)
    return name

def get_users_name(db, user_id):
    usersname = db.get_user_name(user_id)
    return usersname

if __name__ == "__main__":
    db = DB()
    print("****************Welcome to the Task Shop****************")
    print("Please login before continue.")
    username = input("Enter username: ")
    password = input("Enter password: ")
    user = db.get_user(username, password)
    user_role = db.get_role(user)
    while True:
        if user_role[1] == "admin":
            choice = get_admins_option()
            if choice == 1:
                items = db.get_items()
                for x in items:
                    print("Item Name: {} \t Item Price: {}".format(x[1], x[2]))
            if choice == 2:
                item_name = input("Enter product name: ")
                price = float(input("Enter price: "))
                add_product = db.insert_item(item_name, price, user[0])
                if add_product:
                    print("Product added successfully")
            if choice == 3:
                item_id = input("Enter ID to delete item: ")
                response = db.remove_item(item_id)
                if response:
                    print("Item removed successfully!!!!!")
            if choice == 4:
                order_report = db.get_all_orders()
                print("ID \t items \t user \t Total Amount")
                for x in order_report:
                    id = x[0]
                    item_name = get_item_name(db, x[1])
                    usersname = get_users_name(db, x[3])
                    print(x[0], "\t", item_name, "\t", usersname, "\t", x[4])

        else:
            choice = get_users_option()
            if choice == 1:
                items = db.get_items()
                for x in items:
                    print("ID: {} \t Item Name: {} \t Item Price: {}".format(x[0], x[1], x[2]))
            if choice == 2:
                print("***** Add products to cart *****")
                items = [int(x) for x in input("Enter item ID to add it to cart: ").split()]
                add_to_cart = db.add_to_cart(items, user[0])
                if add_to_cart:
                    print("Products added successfully")
            if choice == 3:
                print("***** Book Order *****")
                cart = db.get_cart(user[0])
                total_amount = db.get_order_total(cart[1])
                order = db.create_order(cart[1], cart[0], cart[2], total_amount)
                if order:
                    print("Ordered Successfully!!!!!!!")
            if choice == 4:
                print("***** Your Orders *****")
                orders = db.get_order_history(user[0])
                print("ID \t Items \t Date \t Amount")
                for x in orders:
                    print(x[0], "\t", x[1], "\t", x[3], "\t", x[2])