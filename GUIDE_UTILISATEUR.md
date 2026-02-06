# Guide utilisateur MeshWiki

## Poser une question

Depuis votre radio Meshtastic, envoyez un message commençant par `?` :

```
?Quelle est la capitale de Madagascar ?
```

En message direct au noeud MeshWiki, le `?` est optionnel.

## Format des réponses

Les réponses sont envoyées en un ou plusieurs messages numérotés :

```
[1/2] Antananarivo est la capitale de Madagascar. C'est la plus grande
ville du pays avec environ 1,3 million d'habitants dans la commune...
[2/2] ...et plus de 3 millions dans l'agglomération. Elle est située
dans les Hautes Terres centrales de l'île.
```

Les réponses longues sont tronquées à 5 messages maximum.

## Pendant l'initialisation

Au premier démarrage, MeshWiki indexe Wikipedia (2-5h). Pendant ce temps, vos questions reçoivent quand même une réponse du LLM, mais sans source Wikipedia. Ces réponses se terminent par un avertissement avec le temps restant estimé :

```
[Sans source Wikipedia — base disponible dans environ 1h30]
```

## Limites

- **Débit** : 10 questions par tranche de 10 minutes par noeud. Au-delà, un message indique le temps d'attente.
- **Taille** : les réponses sont concises (adaptées à la transmission radio LoRa).
- **Langue** : les réponses sont en français, basées sur Wikipedia FR.
- **Précision** : les réponses sont générées par un modèle IA à partir d'extraits Wikipedia. Elles peuvent contenir des erreurs.

## Exemples de questions

```
?Qui a inventé la radio ?
?Quels sont les volcans actifs à La Réunion ?
?C'est quoi le protocole TCP/IP ?
?Quelle est la population de Saint-Denis ?
?Comment fonctionne un panneau solaire ?
```

## Conseils

- Les questions courtes et précises donnent de meilleurs résultats.
- Le système recherche dans Wikipedia FR : les sujets bien documentés en français auront de meilleures réponses.
- Pas besoin d'internet : tout fonctionne localement après l'installation initiale.
