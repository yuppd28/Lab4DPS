# Лабораторна робота №4
# Клюшніченко Дарина ІТ

---

### Однорідна система

dx/dt = P(x, y)
dy/dt = Q(x, y)

де `P(x, y)` і `Q(x, y)` — однорідні функції одного степеня.  
Тип особливої точки визначається поведінкою траєкторій поблизу початку координат.  

### Лінійна система
dx/dt = ax + by
dy/dt = cx + dy

Тип точки визначається за власними значеннями матриці:  

A = | a b |
| c d |

---

## Приклади

### 1. Однорідна система (тема 4.1)

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    dx = x**2 - y**2
    dy = 2*x*y
    return dx, dy

Y, X = np.mgrid[-2:2:100j, -2:2:100j]
U, V = f(X, Y)
plt.streamplot(X, Y, U, V, density=1.4)
plt.title("Однорідна система: x' = x² - y², y' = 2xy")
plt.xlabel("x"); plt.ylabel("y"); plt.grid(True)
plt.show()
```

Результат: сідлова точка в центрі.
<img width="1280" height="669" alt="image" src="https://github.com/user-attachments/assets/a7d6a141-c47f-4db7-ac7c-ceccd80c45ec" />

### 2. Лінійна система (тема 4.2)

dx/dt = y
dy/dt = -x

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    dx = y
    dy = -x
    return dx, dy

Y, X = np.mgrid[-2:2:100j, -2:2:100j]
U, V = f(X, Y)
plt.streamplot(X, Y, U, V, density=1.3)
plt.title("Лінійна система: x' = y, y' = -x")
plt.xlabel("x"); plt.ylabel("y"); plt.grid(True)
plt.show()
```
Результат: центр, замкнені траєкторії навколо початку координат.
<img width="1235" height="707" alt="image" src="https://github.com/user-attachments/assets/345b2488-47dd-4e01-86e8-c4589d74f1fd" />
