"""
海龟作图系统 turtle module
爬行：forward(n) ; backword(n)
转向：left(a) ; right(a)
抬笔放笔：penup() ; pendown()
笔属性：pensize(s) ; pencolor(c)
"""

import turtle
# t = turtle.Turtle()
'画一条线'
# t = turtle.Turtle()
# t.forward(100)
# turtle.done()
'画正方形'
# t = turtle.Turtle()
# for i in range(4):
#     t.forward(100)
#     t.right(90)
# turtle.done()
'画五角星'
# t = turtle.Turtle()
# t.pencolor('red')
# t.pensize(3)
# for i in range(5):
#     t.forward(100)
#     t.right(144)
# t.hideturtle()
#
# turtle.done()
'螺旋线'
def drawSpiral(t, linelen):
    if linelen > 0:
        t.forward(linelen)
        t.right(90)
        drawSpiral(t, linelen - 5)
    turtle.done()
# drawSpiral(t, 100)
'分形树'
def tree(n, t):
    if n > 5:
        t.forward(n)
        t.right(20)
        tree(n - 15, t)
        t.left(40)
        tree(n - 15, t)
        t.right(20)
        t.backward(n)

def main():
    t = turtle.Turtle()
    mywin = turtle.Screen()
    t.left(90)
    t.up()
    t.backward(100)
    t.down()
    t.color('green')
    tree(75, t)
    mywin.exitonclick()

main()