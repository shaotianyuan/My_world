def test(n=2):
    for i in range(n):
        print(f'1i:{i}')
        test(n-1)
        print(f'2i:{i}')

    # print('-------')

test()

# for i in range(2):
#     print(i)

for i in range(2):
    # print(i)
    for j in range(1):
        print(j)
    print(i)
