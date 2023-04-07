#!usr/bin/env python
# encoding:utf-8

import os

import openai

openai.api_key = "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"

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", messages=[{"role": "user", "content": "狮子猫是什么样子的"}]
)

print(completion.choices[0].message.content)

response = openai.Image.create(prompt="一只白色异瞳小狮子猫", n=1, size="1024x1024")
print(response["data"][0]["url"])
