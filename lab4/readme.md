Расставляем наугад N центров масс, каждую точку относим к классу ближайшего ЦС. Далее ЦС сдвигаем к новому цс всех точек данного цс.
k-means как работает: [видео](https://youtu.be/R_w7PnKWOgw?si=Kfy0ltaI3B_7ijnx)

Все точки двигаем по кд по направлению умного вектора (в сторону более большого скопления массы в заданном радиусе) такс сказать, пока точка не остановится по сути
Все пути привели в центр масс. Сколько их вышло, столько их и есть.
Алгоритм сам определяет сколько центров масс у нас в отличии от k-means, где ЦС N штук рандомно выбирались, а потом двигались.
mean-shift: https://www.youtube.com/watch?v=dNANpVZnIfA