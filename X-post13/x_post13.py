import turtle

# Turtle nesnesini oluştur
tur = turtle.Turtle()

# Ekranı oluştur
ekran = turtle.Screen()
ekran.setup(width=720, height=420)
ekran.bgcolor("red")

# Beyaz daire çizimi (ay)
tur.up()
tur.goto(-100, -100)
tur.color('white')
tur.begin_fill()
tur.circle(120)
tur.end_fill()

# Kırmızı daire çizimi (ay hilali)
tur.goto(-70, -80)
tur.color('red')
tur.begin_fill()
tur.circle(100)
tur.end_fill()

# Yıldız çizimi
tur.goto(0, 35)
tur.fillcolor('white')
tur.pencolor('white')
tur.pendown()
tur.width(1)
tur.begin_fill()

# Beş köşeli yıldız çizimi
side_length = 50
for _ in range(5):
    tur.forward(side_length)
    tur.left(72)
    tur.forward(side_length)
    tur.right(144)

tur.end_fill()

# Turtle'ı gizle ve tıklama ile ekranı kapat
tur.hideturtle()
ekran.exitonclick()