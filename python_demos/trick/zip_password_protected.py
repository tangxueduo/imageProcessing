# # import time
# import zipfile
# from io import StringIO

# # import numpy as np
# import pandas as pd
# # import pyminizip
# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/")
# def save_data(data: dict):
#     df = pd.DataFrame(data)
#     print(df)
#     # # df.to_csv("test.csv", mode="w")
#     # # -------------------- Method 1 -------
#     # compression_opts = dict(method="zip", archive_name="test.csv")
#     # # # TODO: 确认存储 path， 增删改查操作
#     # df.to_csv("test.zip", index=False, compression=compression_opts)
#     # encrypt_status = encrypt_zip("test.zip")
#     # print(encrypt_status)
#     # ----------------------- Method 2 -------
#     output = StringIO()
#     f = zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED)
#     f.writestr("test.csv", df)
#     f.close()
#     # response = HttpResponse(output.getvalue(), mimetype="application/zip")
#     return response
#     # compression_level = 5
#     # pyminizip.compress("test.csv", None, "test.zip", "tuixiang", compression_level)


# def encrypt_zip(zip_path: str, password: str = "tuixiang") -> str:
#     """password-protected zip 文件
#     # param: zip_path,zip 文件存储 path
#     Returns:
#             200, 500...
#     """
#     try:
#         with zipfile.ZipFile(zip_path, "r") as file:
#             # 'setpassword' method is used to give a password to the 'Zip';
#             # file.setpassword(pwd=bytes(password, encoding="utf-8"))
#             file.extractall(pwd=b"{password}")
#     except Exception as ex:
#         raise ex
#     return "success"


# def decrypt_zip(zip_path: str, password: str = "tuixiang"):
#     """根据密码解压 zip 文件"""
#     pass
