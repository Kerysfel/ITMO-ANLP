from pymilvus import connections, utility

connections.connect(host="localhost", port="19530")

if utility.has_collection("cheat_sheets"):
    print("Подключение к Milvus установлено, коллекция существует.")
else:
    print("Подключение есть, но коллекция не найдена.")