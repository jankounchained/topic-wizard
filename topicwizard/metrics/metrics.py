"""
"""
from typing import Any, List, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline

import topicwizard.prepare.topics as prepare
from topicwizard.app import split_pipeline
from topicwizard.prepare.utils import get_vocab, prepare_transformed_data


def exclusive_vocabulary_ratio(
    corpus: Iterable[str],
    pipeline: Optional[Pipeline] = None,
    vectorizer: Any = None,
    topic_model: Any = None,
    topic_names: Optional[List[str]] = None,
    n_top_words: int = 0,
) -> pd.DataFrame:
    """
    Extent to which words are unique to specific topics. 
    Caluclated as % of top n words in a topic that are not in other topics, per topic.

    It helps identify words that are highly representative of a particular topic and not commonly used in other topics. 
    Higher exclusive vocabulary indicates better topic separation, and this metric can be used for both NMF and LDA.
    """
    vectorizer, topic_model = split_pipeline(vectorizer, topic_model, pipeline)
    if pipeline is None:
        pipeline = make_pipeline(vectorizer, topic_model)
    if topic_names is None:
        topic_names = prepare.infer_topic_names(pipeline)
    corpus = list(corpus)
    (
        document_term_matrix,
        document_topic_matrix,
        topic_term_matrix,
    ) = prepare_transformed_data(vectorizer, topic_model, corpus)

    # exclusive vocabulary ratio

    exclusive_vocabulary_ratios = []
    feature_names = vectorizer.get_feature_names()

    for topic_idx, topic in enumerate(topic_term_matrix):
        other_topics = set(range(len(topic_term_matrix))) - {topic_idx}
        exclusive_words = []

        for other_topic in other_topics:
            other_topic_words = topic_term_matrix[other_topic]
            exclusive_words.extend(
                [feature_names[i] for i in topic.argsort()[:n_top_words] if feature_names[i] not in exclusive_words and feature_names[i] not in feature_names[other_topic_words.argsort()[-n_top_words:]]]
            )

        exclusive_vocabulary_ratio = len(exclusive_words) / n_top_words * 100
        exclusive_vocabulary_ratios.append(exclusive_vocabulary_ratio)

    result_df = pd.DataFrame(
        {"Topic": topic_names, "Exclusive Vocabulary Ratio": exclusive_vocabulary_ratios}
    )

    return result_df


if __name__ == "__main__":
    pass
