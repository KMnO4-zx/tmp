#include <stdio.h>
#include <math.h>

int main() {
    float p, s, r;
    p = 3.14;
    printf("请输入圆的半径: \n");
    scanf("%f", &r);
    s = p * r * r;
    printf("圆的面积是: %.2f\n", s);
    return 0;
}