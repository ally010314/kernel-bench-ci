# 배포할 모델 명세

파일 하나가 모델 하나다. PR 에서 이 폴더의 파일을 바꾸면 `selector.yml` 워크플로가 칩마다
**커스텀 sparse MLP 를 쓸지 · 어떤 타일 설정으로 · E2E 가 얼마나 빨라질지** 를 PR 댓글로 단다.

```yaml
name: vitBclip                     # 표시 이름
timm: vit_base_patch16_clip_224    # export 할 timm 이름
shape: {N: 197, dim: 768, hidden: 3072, blocks: 12}   # 토큰 수 · channel MLP · 블록 수 (Mixer 는 token_hidden 도)
chips: [orin, ada, a6000, rtx5090] # 대상 칩
```

`shape` 를 비우면 `selector/shapes.yml` 에서 `name` 으로 찾는다. 로컬에서는 `python3 -m selector predict models/<파일>.yml`.
