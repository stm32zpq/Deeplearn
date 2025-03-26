# call的使用是可以快速调用class
class   Person:
        def __call__(self,name):
            print("__call"+"hello")

        def hell(self,name):
            print("hello"+name)
person = Person()
person("zhangsan")
person.hell("list")