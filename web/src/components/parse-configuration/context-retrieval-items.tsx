import { useTranslate } from '@/hooks/common-hooks';
import { cn } from '@/lib/utils';
import { Form, Input, InputNumber, Select, Slider, Switch } from 'antd';
import { Flex } from 'antd';
import { useCallback, useMemo } from 'react';
import { DatasetConfigurationContainer } from '../dataset-configuration-container';

type ContextRetrievalItemsProps = {
  marginBottom?: boolean;
};

const enum MethodValue {
  Semantic = 'semantic',
  Hybrid = 'hybrid',
}

const DEFAULT_CONTEXT_PROMPT = `
<document>
{doc_content}
</document>

Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
`;

export function UseContextRetrievalItem() {
  const { t } = useTranslate('knowledgeConfiguration');

  return (
    <Form.Item
      name={['parser_config', 'context_retrieval', 'use_context_retrieval']}
      label={t('useContextRetrieval')}
      initialValue={false}
      valuePropName="checked"
      tooltip={t('useContextRetrievalTip')}
    >
      <Switch />
    </Form.Item>
  );
}

const ContextRetrievalItems = ({ marginBottom = false }: ContextRetrievalItemsProps) => {
  const { t } = useTranslate('knowledgeConfiguration');

  const methodOptions = useMemo(() => {
    return [MethodValue.Semantic, MethodValue.Hybrid].map((x) => ({
      value: x,
      label: x === MethodValue.Semantic ? 'Semantic' : 'Hybrid',
    }));
  }, []);

  const renderWideTooltip = useCallback(
    (title: React.ReactNode | string) => {
      return {
        title: typeof title === 'string' ? t(title) : title,
        overlayInnerStyle: { width: '32vw' },
      };
    },
    [t],
  );

  return (
    <DatasetConfigurationContainer className={cn({ 'mb-4': marginBottom })}>
      <UseContextRetrievalItem></UseContextRetrievalItem>
      <Form.Item
        shouldUpdate={(prevValues, curValues) =>
          prevValues.parser_config.context_retrieval?.use_context_retrieval !==
          curValues.parser_config.context_retrieval?.use_context_retrieval
        }
      >
        {({ getFieldValue }) => {
          const useContextRetrieval = getFieldValue([
            'parser_config',
            'context_retrieval',
            'use_context_retrieval',
          ]);

          return (
            useContextRetrieval && (
              <>
                <Form.Item
                  name={['parser_config', 'context_retrieval', 'method']}
                  label={t('retrievalMethod')}
                  tooltip={renderWideTooltip(
                    <div
                      dangerouslySetInnerHTML={{
                        __html: t('retrievalMethodTip'),
                      }}
                    ></div>,
                  )}
                  initialValue={MethodValue.Semantic}
                >
                  <Select options={methodOptions} />
                </Form.Item>
                <Form.Item
                  name={['parser_config', 'context_retrieval', 'context_prompt']}
                  label={t('contextPrompt')}
                  tooltip={t('contextPromptTip')}
                  initialValue={DEFAULT_CONTEXT_PROMPT}
                >
                  <Input.TextArea rows={12} />
                </Form.Item>
                <Form.Item label={t('windowSize')} tooltip={t('windowSizeTip')}>
                  <Flex gap={20} align="center">
                    <Flex flex={1}>
                      <Form.Item
                        name={['parser_config', 'context_retrieval', 'window_size']}
                        noStyle
                        initialValue={3}
                        rules={[
                          {
                            required: true,
                            message: t('windowSizeMessage'),
                          },
                        ]}
                      >
                        <Slider min={1} max={10} style={{ width: '100%' }} />
                      </Form.Item>
                    </Flex>
                    <Form.Item
                      name={['parser_config', 'context_retrieval', 'window_size']}
                      noStyle
                      rules={[
                        {
                          required: true,
                          message: t('windowSizeMessage'),
                        },
                      ]}
                    >
                      <InputNumber max={10} min={1} />
                    </Form.Item>
                  </Flex>
                </Form.Item>
                {getFieldValue(['parser_config', 'context_retrieval', 'method']) === MethodValue.Hybrid && (
                  <>
                    <Form.Item label={t('hybridSearchWeights')} tooltip={t('hybridSearchWeightsTip')}>
                      <Flex gap={20} align="center">
                        <Flex flex={1} vertical>
                          <span>{t('semanticWeight')}</span>
                          <Form.Item
                            name={['parser_config', 'context_retrieval', 'semantic_weight']}
                            noStyle
                            initialValue={0.8}
                            rules={[
                              {
                                required: true,
                                message: t('weightMessage'),
                              },
                            ]}
                          >
                            <Slider min={0} max={1} step={0.1} style={{ width: '100%' }} />
                          </Form.Item>
                        </Flex>
                        <Flex flex={1} vertical>
                          <span>{t('bm25Weight')}</span>
                          <Form.Item
                            name={['parser_config', 'context_retrieval', 'bm25_weight']}
                            noStyle
                            initialValue={0.2}
                            rules={[
                              {
                                required: true,
                                message: t('weightMessage'),
                              },
                            ]}
                          >
                            <Slider min={0} max={1} step={0.1} style={{ width: '100%' }} />
                          </Form.Item>
                        </Flex>
                      </Flex>
                    </Form.Item>
                  </>
                )}
              </>
            )
          );
        }}
      </Form.Item>
    </DatasetConfigurationContainer>
  );
};

export default ContextRetrievalItems; 