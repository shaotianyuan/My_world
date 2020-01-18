import turtle

def tree(branch):
    if branch > 5:
        t.forward(branch)
        t.right(20)
        tree(branch - 15)
        t.left(40)
        tree(branch - 15)
        t.right(20)
        t.backward(branch)

t = turtle.Turtle()
t.left(90)
t.penup()
t.backward(100)
t.pendown()
t.pencolor('green')
t.pensize(2)
tree(75)
t.hideturtle()
turtle.done()