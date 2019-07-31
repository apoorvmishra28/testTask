import shopdb
import time
import datetime
import mysql.connector

shopdb = shopdb.shopDB()

mydb = shopdb.connect()
mycursor = mydb.cursor()

created_at = time.time()

class DB:
    def create_users(self):
        try:
            mycursor.execute(
                "CRATE TABLE IF NOT EXISTS users (id INT AUTO_INCREMENT PRIMARY KEY, "
                "username VARCHAR(255) NOT NULL UNIQUE, password VARCHAR(255) NOT NULL"
            )
            return True
        except mysql.connector.Error as e:
            print("Error: {}".format(e))

    def create_user_roles(self):
        try:
            mycursor.execute(
                "CRATE TABLE IF NOT EXISTS users_role (id INT AUTO_INCREMENT PRIMARY KEY, "
                "role VARCHAR(255) NOT NULL DEFAULT 'user', user_id INT NOT NULL, FOREIGN KEY (user_id) REFERENCES users(id)"
            )
            return True
        except mysql.connector.Error as e:
            print("Error: {}".format(e))

    def create_items(self):
        try:
            mycursor.execute(
                "CRATE TABLE IF NOT EXISTS items (id INT AUTO_INCREMENT PRIMARY KEY, "
                "item_name VARCHAR(255) NOT NULL, price DECIMAL(10, 7) NOT NULL, "
                "user_id INT NOT NULL, FOREIGN KEY (user_id) REFERENCES users(id)"
            )
            return True
        except mysql.connector.Error as e:
            print("Error: {}".format(e))

    def create_cart(self):
        try:
            mycursor.execute(
                "CRATE TABLE IF NOT EXISTS cart (id INT AUTO_INCREMENT PRIMARY KEY, "
                "item_id INT NOT NULL, FOREIGN KEY (item_id) REFERENCES items(id), "
                "user_id INT NOT NULL, FOREIGN KEY (user_id) REFERENCES users(id)"
            )
            return True
        except mysql.connector.Error as e:
            print("Error: {}".format(e))

    def create_order_table(self):
        try:
            mycursor.execute(
                "CRATE TABLE IF NOT EXISTS orders (id INT AUTO_INCREMENT PRIMARY KEY, "
                "item_id INT NOT NULL, FOREIGN KEY (item_id) REFERENCES items(id),"
                "cart_id INT NOT NULL, FOREIGN KEY (cart_id) REFERENCES cart(id),"
                "user_id INT NOT NULL, FOREIGN KEY (user_id) REFERENCES users(id),"
                "total_amount DECIMAL(10,7) NOT NULL, created_at TIMESTAMP NOT NULL"
            )
            return True
        except mysql.connector.Error as e:
            print("Error: {}".format(e))

    def get_user(self, username, password):
        try:
            mycursor.execute(
                "SELECT * FROM users WHERE username='{}' AND password='{}'".format(
                    username, password
                )
            )
            user = mycursor.fetchone()
            return user
        except mysql.connector.Error as e:
            print("Error: {}".format(e))

    def get_user_name(self, user_id):
        try:
            mycursor.execute(
                "SELECT username FROM users WHERE id='{}'".format(
                    user_id
                )
            )
            user = mycursor.fetchone()
            return user[0]
        except mysql.connector.Error as e:
            print("Error: {}".format(e))

    def get_role(self, user_id):
        try:
            mycursor.execute(
                "SELECT role FROM users_role WHERE user_id='{}'".format(user_id)
            )
            role = mycursor.fetchone()
            return role
        except mysql.connector.Error as e:
            print("Error: {}".format(e))

    def get_order_total(self, item_ids):
        try:
            mycursor.execute(
                "SELECT SUM(price) FROM items WHERE id IN '{}'".format(item_ids)
            )
            price = mycursor.fetchone()
            return price[0]
        except mysql.connector.Error as e:
            print("Error: {}".format(e))

    def get_items(self):
        try:
            mycursor.execute(
                "SELECT * FROM items"
            )
            items = mycursor.fetchall()
            return items
        except mysql.connector.Error as e:
            print("Error: {}".format(e))

    def get_items_name(self, item_id):
        try:
            mycursor.execute(
                "SELECT item_name FROM items WHERE id='{}'".format(item_id)
            )
            name = mycursor.fetchone()
            return name[0]
        except mysql.connector.Error as e:
            print("Error: {}".format(e))

    def get_order_history(self, user_id):
        try:
            mycursor.execute(
                "SELECT (id, item.item_name, total_amount, created_at) FROM orders LEFT JOIN items ON "
                "orders.item_id = item.id WHERE orders.user_id='{}' ORDER BY created_at DESC".
                    format(user_id)
            )
            orders = mycursor.fetchall()
            return orders
        except mysql.connector.Error as e:
            print("Error: {}".format(e))

    def get_all_orders(self):
        try:
            mycursor.execute(
                "SELECT * FROM orders"
            )
            orders = mycursor.fetchall()
            return orders
        except mysql.connector.Error as e:
            print("Error: {}", e)

    def add_user(self, username, password):
        try:
            mycursor.execute(
                "INSERT INTO users(username, password) VALUES('{}', '{}')".
                    format(username, password)
            )
            mycursor.commit()
            return True
        except mysql.connector.Error as e:
            print("Error: {}".format(e))

    def insert_item(self, item_name, price, user_id):
        try:
            mycursor.execute(
                "INSERT INTO items(item_name, price, user_id) VALUES('{}', {}, {})".
                    format(item_name, price, user_id)
            )
            mycursor.commit()
            return True
        except mysql.connector.Error as e:
            print("Error: {}".format(e))

    def remove_item(self, id):
        try:
            mycursor.execute(
                "DELETE FROM items WHERE id = '{}'".format(id)
            )
            mycursor.commit()
            return True
        except mysql.connector.Error as e:
            print("Error: {}".format(e))

    def add_to_cart(self, item_id, user_id):
        try:
            mycursor.execute(
                "INSERT INTO cart(item_id, user_id) VALUES({}, {})".
                    format(item_id, user_id)
            )
            mycursor.commit()
            return True
        except mysql.connector.Error as e:
            print("Error: {}".format(e))

    def get_cart(self, user_id):
        try:
            mycursor.execute(
                "SELECT * FROM carts WHERE user_id = '{}' ORDER BY id DESC LIMIT(1)".format(user_id)
            )
            cart = mycursor.fetchone()
            return cart
        except mysql.connector.Error as e:
            print("Error: {}".format(e))

    def create_order(self, item_id, cart_id, user_id, total_amount):
        global created_at
        created_at = time.time()
        timestamp = datetime.datetime.fromtimestamp(created_at).strftime('%Y-%m-%d %H:%M:%S')
        try:
            mycursor.execute(
                "INSERT INTO orders(item_id, cart_id, user_id, created_at, total_amount) VALUES".
                    format(item_id, cart_id, user_id, timestamp, total_amount)
            )
            mycursor.commit()
            return True
        except mysql.connector.Error as e:
            print("Error: {}".format(e))

    def seed_dummy_users(self):
        try:
            mycursor.execute(
                "INSERT INTO users (username, password) VALUES ('{}', '{}')".
                    format(('admin', 'admin'), ('test1', '123456'), ('test2', '123456'))
            )
            mycursor.commit()
        except mysql.connector.Error as e:
            print("Error: {}".format(e))

    def seed_dummy_roles(self):
        try:
            mycursor.execute(
                "INSERT INTO users_role (role, user_id) VALUES ('{}', {})".
                    format(('admin', 1), ('user', 2), ('user', 3))
            )
            mycursor.commit()
        except mysql.connector.Error as e:
            print("Error: {}".format(e))

    def seed_dummy_items(self):
        try:
            mycursor.execute(
                "INSERT INTO items (item_name, price, user_id) VALUES ('{}', {}, {})".
                    format(('Adidas', 2795.00, 1), ('Fossil', 9780.95, 1), ('Oneplus 7', 42000.00, 1))
            )
            mycursor.commit()
        except mysql.connector.Error as e:
            print("Error: {}".format(e))
    def close_connection(self):
        mycursor.close()
        mydb.close()

db = DB()
db.create_users()
db.create_user_roles()
db.create_items()
db.create_cart()
db.create_order_table()
db.seed_dummy_users()
db.seed_dummy_roles()
db.seed_dummy_items()
