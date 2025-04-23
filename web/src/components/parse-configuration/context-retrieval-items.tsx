import { useTranslate } from '@/hooks/common-hooks';
import { cn } from '@/lib/utils';
import { Form, Input, Switch } from 'antd';
import { useCallback } from 'react';
import { DatasetConfigurationContainer } from '../dataset-configuration-container';

type ContextRetrievalItemsProps = {
  marginBottom?: boolean;
};

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

const ContextRetrievalItems = ({
  marginBottom = false,
}: ContextRetrievalItemsProps) => {
  const { t } = useTranslate('knowledgeConfiguration');

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
                  name={[
                    'parser_config',
                    'context_retrieval',
                    'context_prompt',
                  ]}
                  label={t('contextPrompt')}
                  tooltip={t('contextPromptTip')}
                  initialValue={DEFAULT_CONTEXT_PROMPT}
                >
                  <Input.TextArea rows={12} />
                </Form.Item>
              </>
            )
          );
        }}
      </Form.Item>
    </DatasetConfigurationContainer>
  );
};

export default ContextRetrievalItems;
