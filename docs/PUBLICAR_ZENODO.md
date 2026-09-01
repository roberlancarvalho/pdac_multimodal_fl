# Publicar no Zenodo (arquivamento + DOI)

O Zenodo integra com o GitHub: você liga o repositório uma vez, e **cada
*release*** do GitHub vira um pacote arquivado no Zenodo com **DOI**.

## 0. Antes de começar (no repositório)

Já foram adicionados:
- `CITATION.cff` — metadados de citação (o GitHub mostra "Cite this repository").
- `.zenodo.json` — metadados que o Zenodo usa no lugar dos padrões do GitHub.

**Edite os dois** e preencha o que falta:
- seu **ORCID** (crie em <https://orcid.org> se não tiver) — descomente a linha `orcid:` no `CITATION.cff`;
- confirme autores/afiliação (hoje: você + Josefino Cabral Melo Lima, PPGI/UFRJ — remova o 2º se for só você);
- ajuste `version` e `date-released` para a release que você vai criar.

Faça commit e push dessas mudanças **antes** de criar a release.

```bash
git add CITATION.cff .zenodo.json README.md
git commit -m "(docs): metadados Zenodo (CITATION.cff, .zenodo.json)"
git push
```

## 1. Ligar o repositório no Zenodo

1. Entre em <https://zenodo.org> → **Log in with GitHub** (autorize).
2. Menu do seu nome → **GitHub** (ou <https://zenodo.org/account/settings/github/>).
3. Se o repo `roberlancarvalho/pdac_multimodal_fl` não aparecer, clique em **Sync now**.
4. Vire o **toggle** ao lado do repositório para **ON**.

A partir daqui, toda release nova é capturada automaticamente.

## 2. Criar a release no GitHub

1. No GitHub: **Releases** → **Draft a new release**.
2. **Choose a tag** → digite `v0.1.0` → **Create new tag on publish**.
3. **Release title**: `v0.1.0 — Pipeline Multimodal Federado para PDAC`.
4. Descrição: um resumo (pode colar o do `CITATION.cff` ou o topo do README).
5. **Publish release**.

Em 1–2 min o Zenodo cria o registro. Confira em
<https://zenodo.org/account/settings/github/> (aparece o DOI ao lado do repo) ou
na aba **Upload** do seu Zenodo.

## 3. Revisar e completar os metadados no Zenodo

1. Abra o registro criado → **Edit**.
2. Confira: *Upload type* = **Software**, licença = **MIT**, autores + ORCID +
   afiliação, *keywords*.
3. **Related/alternate identifiers**: adicione
   `is supplement to` → a URL do GitHub;
   e, quando a revisão sistemática for publicada, `is referenced by` → o DOI dela.
4. *Contributors*: pode adicionar o orientador como **Supervisor** se ele não for autor.
5. **Publish** (ou **Save** e publique depois).

> Dica: o Zenodo tem dois DOIs — o **concept DOI** (sempre aponta para a versão
> mais recente, use este para citar o projeto) e o **version DOI** (fixa a v0.1.0).

## 4. Colocar o badge do DOI no README

Copie o *Markdown* do badge que o Zenodo mostra (na página do registro, seção
**Cite as** / **DOI badge**) e substitua o comentário no topo do `README.md`:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

Commit + push.

## 5. Novas versões

É só criar outra release no GitHub (`v0.2.0`, ...). O Zenodo cria automaticamente
uma nova versão sob o mesmo *concept DOI*. Atualize `version`/`date-released` no
`CITATION.cff` antes de cada release.

## 6. Como citar (para a dissertação)

> Carvalho, R. O. de; Lima, J. C. M. *Pipeline Multimodal Federado para PDAC*
> (v0.1.0). Zenodo, 2026. https://doi.org/10.5281/zenodo.XXXXXXX

---

### Alternativa sem GitHub (upload manual)

Se preferir não ligar a integração: em <https://zenodo.org> → **Upload** → **New
upload** → arraste um `.zip` do repositório (`git archive -o pdac.zip HEAD`),
preencha os metadados à mão e **Publish**. Você perde o versionamento
automático, mas ganha o DOI do mesmo jeito.
