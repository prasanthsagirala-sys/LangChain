from typing import TypedDict

class Person(TypedDict):

    name: str 
    age: int 

new_person: Person = {'name':'Ravi','age':35}

print(new_person)

new_person: Person = {'name':'Ravi','age':'35'}

print(new_person)

new_person = Person(name='Prasanth', age=30)
print(new_person)

new_person = Person(name='Prasanth', age='35') #No Error
print(new_person)