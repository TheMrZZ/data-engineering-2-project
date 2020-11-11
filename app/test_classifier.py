import unittest

import classifier


class TestClassifier(unittest.TestCase):
    def test_positive_compound_sentiment(self):
        self.assertEqual(classifier.get_sentiment_from_compound(0.05), 'positive')
        self.assertEqual(classifier.get_sentiment_from_compound(0.1), 'positive')
        self.assertEqual(classifier.get_sentiment_from_compound(1078), 'positive')

    def test_neutral_compound_sentiment(self):
        self.assertEqual(classifier.get_sentiment_from_compound(0.05 - 1e-3), 'neutral')
        self.assertEqual(classifier.get_sentiment_from_compound(0), 'neutral')
        self.assertEqual(classifier.get_sentiment_from_compound(-0.05 + 1e-3), 'neutral')

    def test_negative_compound_sentiment(self):
        self.assertEqual(classifier.get_sentiment_from_compound(-0.05), 'positive')
        self.assertEqual(classifier.get_sentiment_from_compound(-0.1), 'positive')
        self.assertEqual(classifier.get_sentiment_from_compound(-1078), 'positive')

    def test_positive_prediction(self):
        self.assertEqual(classifier.predict('I love mushrooms'), 'positive')
        self.assertEqual(classifier.predict('This movie was nice'), 'positive')
        self.assertEqual(classifier.predict('What a wonderful day'), 'positive')

    def test_neutral_prediction(self):
        self.assertEqual(classifier.predict('I hate mushrooms'), 'negative')
        self.assertEqual(classifier.predict('I love hating people'), 'negative')
        self.assertEqual(classifier.predict('Why are you always so late?'), 'negative')
        self.assertEqual(classifier.predict('I am sorry, you are fired'), 'negative')

    def test_negative_prediction(self):
        self.assertEqual(classifier.predict('Where is the kitchen?'), 'neutral')
        self.assertEqual(classifier.predict('Near the bathroom'), 'neutral')
        self.assertEqual(classifier.predict('I will buy two oranges please'), 'neutral')


if __name__ == '__main__':
    unittest.main()
