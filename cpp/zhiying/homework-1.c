#include <stdio.h>
#include <math.h>

int main() {
    float a, b, c, s, p;

    printf("请输入三角形的三条边长: \n");
    scanf("%f %f %f", &a, &b, &c);

    s = (a + b + c) / 2;
    p = sqrt(s * (s - a) * (s - b) * (s - c));

    // 如果不能构成三角形，面积为0
    if (a + b <= c || a + c <= b || b + c <= a) {
        p = 0;
        printf("不能构成三角形，面积为: %.2f\n", p);
        return 0;
    }

    printf("三角形的面积是: %.2f\n", p);
    return 0;
}