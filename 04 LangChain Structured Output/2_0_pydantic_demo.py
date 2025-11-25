from enum import Flag
from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'Nitish' #default Values
    age: Optional[int] = None #Optional values

student = Student()
print(student) #Default name assigned

student = Student(name='Prasanth', age = '32') #Pydantic is smart enough to check if string 32 can be made int
print(student)

try:
    student = Student(name=32)
    print(student) #Error 
    # pydantic_core._pydantic_core.ValidationError: 1 validation error for Student
    # name
    #   Input should be a valid string [type=string_type, input_value=32, input_type=int]
except Exception as e:
    print(e)

class Student2(BaseModel):
    name: str = 'Nitish'
    age: Optional[int] = None
    email: EmailStr #Builtin Validation

try: 
    student = Student2(email = 'asfdg')
except Exception as e:
    print(e)

class Student3(BaseModel):
    name: str = 'Nitish'
    age: Optional[int] = None
    email: EmailStr #Builtin Validation
    cgpa: float = Field(gt=0, lt=10, default=5, description='A decimal value representing the cgpa of the student')

try: 
    student = Student3(email = 'asfdg@mail.com', cgpa = 12)
except Exception as e:
    print(e)
