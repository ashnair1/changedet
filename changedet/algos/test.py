class Test(object):
    def _decorator(foo):
        # import pdb; pdb.set_trace()
        def magic(self):
            print("start magic")
            foo(self)
            print("end magic")

        return magic

    @_decorator
    def bar(self):
        print("normal call")


test = Test()

test.bar()
