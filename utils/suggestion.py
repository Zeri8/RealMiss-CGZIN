
def generate_suggestion(mask, prob, th=0.3):
    mods = ['T1', 'T1ce', 'T2', 'FLAIR']
    for i in range(4):
        if mask[i] == 0 and prob[i] < th:
            return f"建议补扫 {mods[i]}（贡献 {prob[i]:.1%}）"
    return "无需补扫"