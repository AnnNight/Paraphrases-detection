import sqlalchemy as alch
import psycopg2
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

p_login = "postgres"
p_password = "239"


def execute_query(query):
    global p_login, p_password
    engine = alch.create_engine(f'postgresql+psycopg2://{p_login}:{p_password}@localhost:5432/web-service-vkr')
    # cursor = engine.connect()
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # session.add(cursor.execute(query))
        res = session.execute(text(query))
        print("Query executed successfully")
        # session.add(res)
        session.commit()
        return res
    except psycopg2.OperationalError as e:
        print(f"The error '{e}' occurred")
    session.close()


def insert_into_user(data: list) -> int:
    print(data)
    query = f"SELECT add_user(\'{data[0]}\', \'{data[1]}\', \'{data[2]}\', \'{data[3]}\');"
    print(query)
    res = execute_query(query)
    rows = res.fetchall()
    print(rows[0][0])
    return rows[0][0]


def get_user(login: str, password: str) -> int:
    query = f'SELECT get_user(\'{str(login)}\', \'{str(password)}\');'
    print(query)
    res = execute_query(query)
    rows = res.fetchall()
    if len(rows) == 0 or None in rows[0]:
        return -1
    else:
        return rows[0][0]


def find_login(login: str) -> int:
    query = f'SELECT find_login(\'{str(login)}\');'
    print(query)
    res = execute_query(query)
    rows = res.fetchall()
    if len(rows) == 0 or None in rows[0]:
        return -1
    else:
        return rows[0][0]


def create_user(login: str, password: str, prhone_number: str, email: str) -> int:
    query = f'SELECT add_user(\'{str(login)}\', \'{str(password)}\', \'{int(prhone_number)}\', \'{str(email)}\');'
    print(query)
    res = execute_query(query)
    rows = res.fetchall()
    if len(rows) == 0 or None in rows[0]:
        return -1
    else:
        return rows[0][0]
