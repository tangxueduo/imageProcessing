import requests

url = "http://172.16.6.8:3333/series/1.3.12.2.1107.5.1.4.76346.30000021052006532895300034737/tasks"
tasks = requests.get(url).json()
print(f"**reverse tasks: {tasks[::-1]}")
res = list(filter(lambda x: x.get("type") == "predict/ct_heart_ffr", tasks))
print(res)
